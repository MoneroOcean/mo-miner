// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

"use strict";

const path = require("node:path");
const {EventEmitter} = require("node:events");
const clusterModule = require("node:cluster");
// @types/node models cluster as a default export, while CommonJS returns that object directly.
const cluster = clusterModule.default ?? clusterModule;
const fs = require("node:fs");
const childProcess = require("node:child_process");
const gpuTuning = require("./gpu-tuning");

/** @typedef {import("node:child_process").ChildProcessWithoutNullStreams} SubprocessWorker */
/** @typedef {import("node:cluster").Worker} ClusterWorker */
/** @typedef {{type: "subprocess", id: number, worker: SubprocessWorker} | {type: "cluster", id: number, worker: ClusterWorker}} WorkerTarget */

const isWindowsProcess = process.platform === "win32";
const developmentBuildPlatform = isWindowsProcess ? "win" : "lin";
const isExplicitWorker = process.env["MOM_CLUSTER_WORKER"] === "1";
const isWorkerProcess = isExplicitWorker ||
  (!isWindowsProcess && !cluster.isPrimary);
// The controller picks a backend left unset (workers inherit it) before anything reads it, starting
// with the AMD worker choice below. An explicitly empty value still selects no backend: native
// parity harnesses rely on that.
if (!isWorkerProcess && process.env["MOM_GPU_BACKEND"] === undefined) {
  const detected = require("./helper/gpu-detect").detectGpuBackend();
  if (detected) {process.env["MOM_GPU_BACKEND"] = detected;}
}
const useSubprocessWorkers = isWindowsProcess ||
  process.env["MOM_USE_SUBPROCESS_WORKERS"] === "1" ||
  (process.env["MOM_GPU_BACKEND"] || "").toLowerCase() === "amd";

/** @returns {number | "master"} */
function processThreadId() {
  if (!isWorkerProcess) {return "master";}
  const value = process.env["thread_id"] || "";
  const id = Number(value);
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(id)) {
    throw new Error(`Invalid worker thread id: ${value || "<missing>"}`);
  }
  return id;
}

const threadId = processThreadId();
/** @type {number[]} */
let workerIds = []; // active worker ids (cluster.workers can contain not yet closed workers)
/** @type {Record<number, SubprocessWorker>} */
let workerProcs = {};
let workerGeneration = 0;
/** @type {WeakMap<SubprocessWorker | ClusterWorker, "requested" | "forced" | "reported">} */
const expectedWorkerCloses = new WeakMap();
/** @type {NativeCoreModule | null} */
let coreModuleForExit = null;
let workerLogLevel = 0;
const WORKER_MESSAGE_PREFIX = "MOM_WORKER_MESSAGE ";
const MAX_WORKER_MESSAGE_LINE = 1024 * 1024;
const {
  MAX_PEARL_PROOF_BASE64, MAX_PROOF_EVENT_OVERHEAD, isIdentifier, isJobNonce, isNonceHex,
} = require("./helper/worker-protocol");
const MAX_WORKER_PROOF_LINE = MAX_PEARL_PROOF_BASE64 + MAX_PROOF_EVENT_OVERHEAD;
const {filterWorkerStderr, filterWorkerStdoutLine} = require("./helper/diagnostics");
const {isObject, errorMessage} = require("./helper/validate");

const miningJobStringFields = [
  "backend", "backend_request", "blob", "extra_nonce", "extranonce2", "header_hash",
  "id", "nicehash_mask", "ntime", "pearlhash_base_target", "pre_pow", "seed_hex", "seed_hash",
  "solution", "submit_mode", "target", "xn", "job_token",
];
const miningJobUnsignedFields = [
  "extra_nonce2_size", "height", "nonce1_len", "nonceoffset", "thread_id",
];
const miningJobPositiveFields = [
  "intensity", "noncebytes", "pearlhash_k", "pearlhash_n",
  "pearlhash_rank", "proofsize", "thread_num",
];

/** @param {unknown} value @returns {value is NativeCoreModule} */
function isNativeCoreModule(value) {
  return isObject(value) && typeof value["AsyncWorker"] === "function" &&
    (value["exitNow"] === undefined || typeof value["exitNow"] === "function");
}

/** @param {unknown} value @returns {value is NativeCoreWorker} */
function isNativeCoreWorker(value) {
  return isObject(value) && typeof value["sendToCpp"] === "function";
}

// Admit only a complete, well-typed job before it crosses into a worker or the native core. Pearl
// shapes are validated as a whole because the native kernel trusts their derived dimensions.
/** @param {unknown} value @returns {value is NativeJob} */
function isNativeJob(value) {
  if (!isObject(value) || typeof value["algo"] !== "string" ||
      typeof value["dev"] !== "string" || typeof value["blob_hex"] !== "string" ||
      !/^(?:[0-9a-f]{2})+$/i.test(value["blob_hex"])) {
    return false;
  }
  for (const field of miningJobStringFields) {
    const fieldValue = value[field];
    if (fieldValue !== undefined && typeof fieldValue !== "string") {return false;}
  }
  if (value["difficulty"] !== undefined &&
      (typeof value["difficulty"] !== "number" || !Number.isFinite(value["difficulty"]) ||
        value["difficulty"] <= 0)) {
    return false;
  }
  for (const field of miningJobUnsignedFields) {
    const fieldValue = value[field];
    if (fieldValue !== undefined &&
        (typeof fieldValue !== "number" || !Number.isSafeInteger(fieldValue) || fieldValue < 0)) {
      return false;
    }
  }
  for (const field of miningJobPositiveFields) {
    const fieldValue = value[field];
    if (fieldValue !== undefined &&
        (typeof fieldValue !== "number" || !Number.isSafeInteger(fieldValue) || fieldValue <= 0)) {
      return false;
    }
  }
  const certVersion = value["pearlhash_cert_version"];
  if (certVersion !== undefined && certVersion !== 3) {
    return false;
  }
  if (value["algo"] === "pearlhash") {
    const pearlShapeFields = ["pearlhash_n", "pearlhash_k", "pearlhash_rank"];
    const hasPearlShape = pearlShapeFields.some((field) => value[field] !== undefined);
    const baseTarget = value["pearlhash_base_target"];
    if (baseTarget !== undefined &&
        (typeof baseTarget !== "string" || !/^(?:0x)?[0-9a-f]{1,64}$/i.test(baseTarget))) {
      return false;
    }
    if (hasPearlShape || baseTarget !== undefined) {
      if (value["intensity"] === undefined ||
          pearlShapeFields.some((field) => value[field] === undefined)) {
        return false;
      }
      try {
        gpuTuning.validatePearlHashShape(
          value["intensity"], value["pearlhash_n"], value["pearlhash_k"],
          value["pearlhash_rank"]);
      } catch {
        return false;
      }
    }
  }
  if (value["nonce"] !== undefined && !isJobNonce(value["nonce"])) {return false;}
  if (value["nicehash_mask"] !== undefined && !isNonceHex(value["nicehash_mask"])) {return false;}
  return ["job_id", "pool_id", "worker_id"].every((field) =>
    value[field] === undefined || isIdentifier(value[field]));
}

/** @param {unknown} value @returns {value is WorkerCommand} */
function isWorkerCommand(value) {
  if (!isObject(value)) {return false;}
  const type = value["type"];
  if (type === "close" || type === "pause") {return value["job"] === undefined;}
  if ((type !== "bench" && type !== "job" && type !== "test") || !isNativeJob(value["job"])) {
    return false;
  }
  if (type !== "job") {return true;}
  const job = value["job"];
  return typeof job.target === "string" && /^(?:0x)?[0-9a-f]{1,64}$/i.test(job.target) &&
    isIdentifier(job.job_id) && isIdentifier(job.pool_id) && isIdentifier(job.worker_id) &&
    typeof job.job_token === "string" && job.job_token.length > 0;
}

/** @param {unknown} value @returns {value is WorkerEvent} */
function isWorkerEvent(value) {
  return Boolean(isObject(value) && isObject(value["value"]) && typeof value["type"] === "string" &&
    typeof value["thread_id"] === "number" && Number.isSafeInteger(value["thread_id"]) &&
    value["thread_id"] >= 0);
}
module.exports.is_worker_event = isWorkerEvent;

// Exit on a later turn so the current handler can finish logging. Unlike the controller's exit, a
// worker does not wait for its output streams to drain.
/** @param {number} code */
function reallyExit(code) {
  setImmediate(() => module.exports.exit_now(code));
}

/** @param {NodeJS.ProcessEnv} extra */
function childEnv(extra) {
  const env = {...process.env, ...extra};
  // Native application workers are self-contained; preserve their supplied loader environment.
  return process.platform === "win32" && globalThis.__momNativeCore === undefined ? withWindowsWorkerPath(env) : env;
}

/** @param {NodeJS.ProcessEnv} env */
function normalizeWindowsPathKey(env) {
  // Windows env var names are case-insensitive; collapse any stray PATH variants
  // (e.g. "Path" and "PATH") onto a single canonical key.
  const pathKey = Object.keys(env).find((key) => key.toLowerCase() === "path") || "Path";
  for (const key of Object.keys(env)) {
    if (key !== pathKey && key.toLowerCase() === "path") {delete env[key];}
  }
  return pathKey;
}

/** @param {NodeJS.ProcessEnv} env */
function withWindowsWorkerPath(env) {
  const appDir = path.dirname(process.execPath);
  return withWindowsPathEntries(env, [
    env["MOM_NATIVE_PATH"] && path.dirname(env["MOM_NATIVE_PATH"]),
    appDir,
    path.join(appDir, "mom"),
    process.cwd(),
    path.join(process.cwd(), "mom"),
    path.join(__dirname, "build", developmentBuildPlatform, "Release"),
  ]);
}

/** @param {NodeJS.ProcessEnv} env @param {(string | undefined)[]} entries */
function withWindowsPathEntries(env, entries) {
  const pathKey = normalizeWindowsPathKey(env);
  const pathValue = env[pathKey] || "";
  env[pathKey] = [...entries, pathValue]
    .filter((entry) => typeof entry === "string" && entry !== "")
    .join(path.delimiter);
  return env;
}

/** @param {(string | undefined)[]} paths @returns {string} */
function firstExistingPath(paths) {
  const found = paths.find((filePath) => filePath !== undefined && fs.existsSync(filePath));
  if (found) {return found;}
  const fallback = paths.at(-1);
  if (fallback) {return fallback;}
  throw new Error("No compute-core path configured");
}

/** @param {string} str */
function debugStartup(str) {
  if (process.env["MOM_DEBUG_STARTUP"]) {console.error("MOM_DEBUG_STARTUP " + str);}
}

/** @param {string} current @param {string | Buffer} chunk @param {number} [limit] */
function appendRecentText(current, chunk, limit = 8192) {
  const next = current + chunk.toString("utf8");
  return next.length > limit ? next.slice(next.length - limit) : next;
}

const WORKER_REPLACE_CLOSE_GRACE_MS = 3000;

/** @param {string} str */
function timestamped(str) {
  return (new Date().toISOString().replace(/T/, " ").replace(/\..+/, "")) + " " + str;
}

function logLevel() {
  return isWorkerProcess ? workerLogLevel : global.opt.log_level;
}

/** @param {string} str */
module.exports.log = function(str) {
  console.log(timestamped(logLevel() >= 1 ? "[0] " + str : str));
};

/** @param {number} level */
function makeLevelLogger(level) {
  return function(/** @type {string} */ str) {
    if (logLevel() >= level) {console.log(timestamped("[" + level + "] " + str));}
  };
}

const redactedComputeMessageKeys = new Set(["plain_proof", "worker_id", "login", "pass", "wallet"]);

/** @param {unknown} value @returns {string} */
function formatComputeMessage(value) {
  try {
    const serialized = JSON.stringify(value, (key, nested) =>
      redactedComputeMessageKeys.has(key) ? "<redacted>" : nested);
    return serialized === undefined ? "undefined" : serialized;
  } catch {
    return "<unprintable compute message>";
  }
}

module.exports.log1 = makeLevelLogger(1);
module.exports.log2 = makeLevelLogger(2);
module.exports.log3 = makeLevelLogger(3);

/** @param {string} str */
module.exports.log_err = function(str) {
  console.error(timestamped("ERROR: " + str));
};

/** Resolve the addon and prepare only its matching Windows loader directory.
 * @param {NodeJS.ProcessEnv} [env]
 * @param {NodeJS.Platform} [platform]
 * @returns {string}
 */
module.exports.prepare_native_core_path = function(env = process.env, platform = process.platform) {
  const appDir = path.dirname(process.execPath);
  const platformBuild = platform === "win32" ? "win" : "lin";
  // The selected addon and runtime are one choice; a missing addon must not load a different worker.
  const corePath = env["MOM_NATIVE_PATH"] || firstExistingPath([
    path.join(appDir, "libs", "mom.node"),
    path.join(appDir, "mom.node"),
    path.join(appDir, "mom", "mom.node"),
    platform === "win32" ? path.join(appDir, "build", "win", "compilers", "oneapi", "mom.node") : undefined,
    path.join(appDir, "build", platformBuild, "Release", "mom.node"),
    path.join(process.cwd(), "libs", "mom.node"),
    path.join(process.cwd(), "mom.node"),
    path.join(process.cwd(), "mom", "mom.node"),
    path.join(__dirname, "libs", "mom.node"),
    path.join(__dirname, "mom.node"),
    platform === "win32" ? path.join(__dirname, "build", "win", "compilers", "oneapi", "mom.node") : undefined,
    path.join(__dirname, "build", platformBuild, "Release", "mom.node"),
  ]);
  if (platform === "win32") {withWindowsPathEntries(env, [path.dirname(corePath)]);}
  return corePath;
};

// Load the native addon (or the native application's linked core) and wrap its AsyncWorker as an
// emitter of core events plus emit_to(). Native commands take string-valued payloads only.
/** @returns {ComputeCore} */
module.exports.create_core = function() {
  this.log3("Starting compute core in " + threadId + " thread");
  /** @type {unknown} */
  let loadedModule = globalThis.__momNativeCore;
  if (loadedModule === undefined) {
    const corePath = module.exports.prepare_native_core_path();
    debugStartup("requiring " + corePath);
    loadedModule = require(corePath);
  }
  if (!isNativeCoreModule(loadedModule)) {throw new Error("Invalid native compute-core module");}
  const coreModule = loadedModule;
  debugStartup("required native module");
  coreModuleForExit = coreModule;
  const emitter = new EventEmitter();
  debugStartup("constructing AsyncWorker");
  const worker = new coreModule.AsyncWorker(
    function(/** @type {string} */ name, /** @type {unknown} */ value) {
      module.exports.log3("Getting from compute core " + threadId + " " + name + " message: " +
                          formatComputeMessage(value));
      emitter.emit(name, value);
    },
    function() { emitter.emit("close"); },
    function(/** @type {Error} */ error) { emitter.emit("error", error); }
  );
  if (!isNativeCoreWorker(worker)) {throw new Error("Invalid native compute-core worker");}
  debugStartup("constructed AsyncWorker");
  return {
    from:    emitter,
    emit_to: function(/** @type {string} */ name, /** @type {object | undefined} */ data) {
      module.exports.log3("Sending to compute core " + threadId + " " + name + " message: " +
                          formatComputeMessage(data));
      /** @type {Record<string, string>} */
      const payload = {};
      // native core expects string values; map null/undefined to empty string
      for (const [key, value] of Object.entries(data || {})) {
        payload[key] = value === undefined || value === null ? "" : String(value);
      }
      debugStartup("sending " + name + " to native module");
      worker.sendToCpp(name, payload);
      debugStartup("sent " + name + " to native module");
    }
  };
};

/** @param {number} code */
module.exports.exit_now = function(code) {
  if (coreModuleForExit && coreModuleForExit.exitNow) {
    coreModuleForExit.exitNow(code);
  }
  process.exit(code);
};

/** @param {string} type @param {unknown} value */
function sendWorkerMessage(type, value) {
  const msg = {type, value, thread_id: threadId};
  if (process.send) {return process.send(msg);}
  return process.stdout.write(WORKER_MESSAGE_PREFIX + JSON.stringify(msg) + "\n");
}

/** @param {ComputeCore} computeCore */
function forwardCoreMessages(computeCore) {
  for (const name of ["test", "last_nonce", "result", "hashrate", "algo_params"]) {
    computeCore.from.on(name, function(v) { sendWorkerMessage(name, v); });
  }
  computeCore.from.on("error", function(error) {
    sendWorkerMessage("error", {message: errorMessage(error)});
  });
}

// The first signal asks the core to close cleanly; a second one exits immediately.
/** @param {() => void} closeWorkerProcess */
function installWorkerExitHandlers(closeWorkerProcess) {
  let receivedSignal = false;
  const receiveSignal = function() {
    if (receivedSignal) {return reallyExit(0);}
    receivedSignal = true;
    closeWorkerProcess();
  };
  process.on("SIGINT", receiveSignal);
  process.on("SIGTERM", receiveSignal);
  if (process.platform === "win32") {
    process.on("SIGBREAK", receiveSignal);
  } else {
    process.on("SIGHUP", receiveSignal);
  }
}

/** @param {ComputeCore} computeCore @param {WorkerJobCommand} msg @param {number} workerThreadId */
function startWorkerJob(computeCore, msg, workerThreadId) {
  // Worker topology is owned at dispatch; capture the count before per-worker device selection/tuning.
  const workerThreadCount = module.exports.get_dev_threads(msg.job.dev);
  // find dev for this specific thread from msg.job.dev list
  const selected = gpuTuning.parseDeviceEntry(
    module.exports.get_thread_dev(workerThreadId, msg.job.dev), msg.job.algo);
  gpuTuning.applyNativeJobTuning(msg.job, selected, msg.job.algo);
  msg.job.thread_id = workerThreadId;
  msg.job.thread_num = workerThreadCount;
  computeCore.emit_to(msg.type, msg.job);
}

/** @param {ComputeCore} computeCore @param {WorkerCommand} msg @param {number} workerThreadId */
function handleWorkerMessage(computeCore, msg, workerThreadId) {
  if (msg.type === "job" || msg.type === "bench" || msg.type === "test") {
    return startWorkerJob(computeCore, msg, workerThreadId);
  }
  return computeCore.emit_to(msg.type);
}

// Subprocess workers receive newline-delimited JSON commands on stdin. Any oversized, malformed,
// or rejected line is fatal for the worker; a clean EOF closes it like a close command.
/** @param {(message: unknown) => boolean} handleMessage @param {() => void} onEnd */
function readStdinMessages(handleMessage, onEnd) {
  let input = "";
  let failed = false;
  const fail = function(/** @type {string} */ message) {
    if (failed) {return;}
    failed = true;
    input = "";
    process.stdin.pause();
    module.exports.log_err(message);
    reallyExit(1);
  };
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", function(chunk) {
    if (failed) {return;}
    input += chunk;
    let eol;
    while ((eol = input.indexOf("\n")) !== -1) {
      const line = input.slice(0, eol);
      input = input.slice(eol + 1);
      if (line.length > MAX_WORKER_MESSAGE_LINE) {
        return fail("Worker IPC input exceeded the message line limit");
      }
      if (!line) {continue;}
      let message;
      try {
        message = JSON.parse(line);
      } catch {
        return fail("Worker IPC input was malformed");
      }
      if (!handleMessage(message)) {
        return fail("Worker IPC input was malformed");
      }
    }
    if (input.length > MAX_WORKER_MESSAGE_LINE) {
      return fail("Worker IPC input exceeded the message line limit");
    }
  });
  process.stdin.once("end", function() {
    if (failed) {return;}
    if (input.trim()) {return fail("Worker IPC input was malformed");}
    onEnd();
  });
}

// In a worker process, own one compute core, forward its events to the controller, and execute
// controller commands until close. Returns false in the controller process.
module.exports.cluster_process = function() {
  if (!isWorkerProcess) {return false;}
  if (typeof threadId !== "number") {throw new Error("Worker process has no numeric thread id");}
  const workerThreadId = threadId;

  // process worker thread env vars
  workerLogLevel = Number.parseInt(process.env["log_level"] || "0", 10);

  const computeCore = this.create_core();
  let exiting = false;
  const closeWorkerProcess = function() {
    if (exiting) {return;}
    exiting = true;
    computeCore.emit_to("close");
  };

  // send message from worker thread to master thread
  forwardCoreMessages(computeCore);
  computeCore.from.on("close",       function()  {
    exiting = true;
    process.exitCode = 0;
    // A cluster disconnect marks shutdown intentional; raw process.disconnect invokes Node's forced exit.
    // Let the event loop drain so the native SYCL cleanup hook runs.
    if (cluster.worker) {
      cluster.worker.disconnect();
    } else if (process.disconnect) {
      process.disconnect();
    } else {
      process.stdin.destroy();
    }
  });

  installWorkerExitHandlers(closeWorkerProcess);
  const handleMessage = function(/** @type {unknown} */ msg) {
    if (!isWorkerCommand(msg)) {return false;}
    if (msg.type === "close") {
      closeWorkerProcess();
      return true;
    }
    if (exiting) {return true;}
    handleWorkerMessage(computeCore, msg, workerThreadId);
    return true;
  };

  // process messages from the master thread
  process.on("message", function(msg) {
    if (handleMessage(msg)) {return;}
    module.exports.log_err("Worker IPC input was malformed");
    reallyExit(1);
  });
  if (!process.send) {readStdinMessages(handleMessage, closeWorkerProcess);}

  return true;
};

// get thread dev stripping ^thread specification from it
/** @param {string} devPart */
function parseThreadDev(devPart) {
  const parsed = gpuTuning.parseDeviceEntry(devPart);
  const processSuffix = parsed.processes > 1 ? `^${parsed.processes}` : "";
  return {
    // Keep *B intact here: its primary-field meaning is algorithm-specific and
    // is resolved in the worker, where the job's algorithm is available.
    dev: devPart.slice(0, devPart.length - processSuffix.length),
    threads: parsed.processes,
  };
}

/** @param {number} threadId @param {string} devs */
module.exports.get_thread_dev = function(threadId, devs) {
  let threadCount = 0;
  for (const devPart of devs.split(",")) {
    const parsed = parseThreadDev(devPart);
    threadCount += parsed.threads;
    if (threadId < threadCount) {return parsed.dev;}
  }
  this.log_err("Can't find " + threadId + " thread device in " + devs + " specification");
  return null;
};

// return number of ^threads in dev specification
/** @param {string} dev */
module.exports.get_dev_threads = function(dev) {
  let threadCount = 0;
  for (const devPart of dev.split(",")) {threadCount += parseThreadDev(devPart).threads;}
  return threadCount;
};

// return dev *batch value
/** @param {string} dev */
module.exports.get_dev_batch = function(dev) {
  try {
    const tuning = gpuTuning.parseDeviceEntry(dev).tuning;
    return tuning.intensity || tuning.m || 1;
  } catch {
    return 1;
  }
};

/** @param {SubprocessWorker | ClusterWorker} worker @param {WorkerCommand} msg */
function markExpectedClose(worker, msg) {
  if (msg.type === "close" && !expectedWorkerCloses.has(worker)) {
    expectedWorkerCloses.set(worker, "requested");
  }
}

/** @param {SubprocessWorker | ClusterWorker} worker @param {WorkerCommand} msg */
function isUnexpectedSendError(worker, msg) {
  return msg.type !== "close" && !expectedWorkerCloses.has(worker);
}

/**
 * @param {number} workerId
 * @param {SubprocessWorker | undefined} worker
 * @param {WorkerCommand} msg
 * @returns {WorkerTarget | null}
 */
function sendSubprocessWorker(workerId, worker, msg) {
  if (!worker) {return null;}
  markExpectedClose(worker, msg);
  /** @type {WorkerTarget} */
  const target = {type: "subprocess", id: workerId, worker};
  if (!worker.stdin || !worker.stdin.writable) {return msg.type === "close" ? target : null;}
  worker.stdin.write(JSON.stringify(msg) + "\n");
  return target;
}

/** @param {ClusterWorker} clusterWorker @param {WorkerCommand} msg @param {Error} error */
function emitClusterSendError(clusterWorker, msg, error) {
  if (isUnexpectedSendError(clusterWorker, msg)) {clusterWorker.emit("error", error);}
}

/** @param {ClusterWorker} clusterWorker */
function canSendClusterWorker(clusterWorker) {
  return !clusterWorker.isConnected || clusterWorker.isConnected();
}

/** @param {ClusterWorker} clusterWorker @param {WorkerCommand} msg */
function sendClusterMessage(clusterWorker, msg) {
  try {
    clusterWorker.send(msg, function(error) {
      if (error) {emitClusterSendError(clusterWorker, msg, error);}
    });
  } catch (error) {
    emitClusterSendError(clusterWorker, msg,
      error instanceof Error ? error : new Error(errorMessage(error)));
    return false;
  }
  return true;
}

/**
 * @param {number} workerId
 * @param {ClusterWorker | undefined} clusterWorker
 * @param {WorkerCommand} msg
 * @returns {WorkerTarget | null}
 */
function sendClusterWorker(workerId, clusterWorker, msg) {
  if (!clusterWorker) {return null;}
  markExpectedClose(clusterWorker, msg);
  /** @type {WorkerTarget} */
  const target = {type: "cluster", id: workerId, worker: clusterWorker};
  if (!canSendClusterWorker(clusterWorker) || !sendClusterMessage(clusterWorker, msg)) {
    return msg.type === "close" ? target : null;
  }
  return target;
}

/** @param {WorkerCommand} msg @returns {WorkerTarget[]} */
module.exports.messageWorkers = function(msg) {
  if (!isWorkerCommand(msg)) {throw new Error("Invalid worker command");}
  /** @type {WorkerTarget[]} */
  const targets = [];
  for (const workerId of workerIds) {
    const target = sendSubprocessWorker(workerId, workerProcs[workerId], msg) ||
                   sendClusterWorker(workerId, cluster.workers?.[workerId], msg);
    if (target) {targets.push(target);}
  }
  return targets;
};

/** @param {SubprocessWorker} worker */
function isSubprocessClosed(worker) {
  return worker.exitCode !== null || worker.signalCode !== null || worker.killed;
}

/** @param {SubprocessWorker} worker @param {NodeJS.Signals} [signal] */
function killProcessTree(worker, signal = "SIGKILL") {
  if (!isWindowsProcess || !worker.pid || globalThis.__momNativeCore !== undefined) {
    worker.kill(signal);
    return false;
  }
  const killer = childProcess.spawn("taskkill", ["/pid", String(worker.pid), "/t", "/f"], {
    stdio: "ignore",
  });
  killer.on("error", function() { worker.kill(signal); });
  return true;
}

/** @param {WorkerTarget} target */
function forceCloseWorker(target) {
  if (target.type === "subprocess") {
    const worker = target.worker;
    if (!isSubprocessClosed(worker)) {
      expectedWorkerCloses.set(worker, "forced");
      killProcessTree(worker);
    }
  } else {
    const worker = target.worker;
    if (worker.isDead && worker.isDead()) {return;}
    expectedWorkerCloses.set(worker, "forced");
    worker.kill("SIGKILL");
  }
}

/** @param {number | null | undefined} forceAfterMs */
module.exports.closeWorkers = function(forceAfterMs) {
  const targets = module.exports.messageWorkers({type: "close"});
  if (forceAfterMs != null) {
    setTimeout(function() {
      for (const target of targets) {forceCloseWorker(target);}
    }, forceAfterMs).unref();
  }
  return targets;
};

/** @param {WorkerMessageHandler} messageHandler @param {number} threadId @param {string} message */
function workerError(messageHandler, threadId, message) {
  messageHandler({
    type: "error",
    value: {message, fatal: true},
    thread_id: threadId,
  });
}

/**
 * @param {number} threadId
 * @param {number | null} code
 * @param {string | null} signal
 * @param {string[]} [detail]
 */
function workerExitMessage(threadId, code, signal, detail = []) {
  return "Worker " + threadId + " exited unexpectedly" +
    (signal ? " with signal " + signal : " with code " + code) +
    (detail.length ? ". " + detail.join(" | ") : "");
}

/**
 * @param {SubprocessWorker | ClusterWorker} worker
 * @param {number} workerId
 * @param {number | null} code
 * @param {string | null} signal
 * @param {string[]} [detail]
 * @returns {boolean}
 */
function handleExpectedWorkerExit(worker, workerId, code, signal, detail = []) {
  const reason = expectedWorkerCloses.get(worker);
  if (reason === "requested") {
    if (code !== 0 || signal !== null) {
      module.exports.log_err(workerExitMessage(workerId, code, signal, detail));
      process.exitCode = 1;
      expectedWorkerCloses.set(worker, "reported");
    }
    return true;
  }
  return reason === "forced" || reason === "reported";
}

/** @param {WorkerMessageHandler} messageHandler @param {number} generation */
function forWorkerGeneration(messageHandler, generation) {
  return function(/** @type {WorkerEvent} */ message) {
    if (generation === workerGeneration) {
      messageHandler(message);
    }
  };
}

// Spawn worker i as a child process (Windows, AMD, or MOM_USE_SUBPROCESS_WORKERS). Its stdout
// interleaves prefixed JSON events with ordinary output: events are size-limited and validated, and
// any protocol violation kills the worker and reports one fatal error. Recent plain output is kept
// only to explain an unexpected exit; IPC payloads never enter that diagnostic.
/** @param {number} i @param {NodeJS.ProcessEnv} env @param {WorkerMessageHandler} messageHandler */
function createSubprocessThread(i, env, messageHandler) {
  const thread = childProcess.spawn(
    process.execPath, process.argv.slice(1), {
      env: {...env, MOM_CLUSTER_WORKER: "1"},
      stdio: ["pipe", "pipe", "pipe"],
    });
  let output = "";
  let recentStdout = "";
  let recentStderr = "";
  let pendingStderr = "";
  let failureReported = false;
  function failWorker(/** @type {string} */ message) {
    if (failureReported || expectedWorkerCloses.has(thread)) {return;}
    failureReported = true;
    expectedWorkerCloses.set(thread, "reported");
    workerError(messageHandler, i, message);
    killProcessTree(thread);
  }
  thread.stdout.setEncoding("utf8");
  thread.stdout.on("data", function(chunk) {
    output += chunk;
    let eol;
    while ((eol = output.indexOf("\n")) !== -1) {
      const line = output.slice(0, eol);
      output = output.slice(eol + 1);
      if (line.startsWith(WORKER_MESSAGE_PREFIX)) {
        if (failureReported) {continue;}
        let message;
        try {
          if (line.length > MAX_WORKER_PROOF_LINE) {throw new Error("oversized worker output");}
          message = JSON.parse(line.slice(WORKER_MESSAGE_PREFIX.length));
        } catch {
          failWorker("Worker " + i + " emitted malformed worker output");
          output = "";
          break;
        }
        if (!isWorkerEvent(message) || message.thread_id !== i) {
          failWorker("Worker " + i + " emitted invalid worker output");
          continue;
        }
        if (line.length > MAX_WORKER_MESSAGE_LINE) {
          const proof = message.value["plain_proof"];
          // Only proof-bearing results get the larger budget; all other envelope fields stay small.
          if (message.type !== "result" || typeof proof !== "string" ||
              proof.length > MAX_PEARL_PROOF_BASE64 ||
              line.length - proof.length > MAX_PROOF_EVENT_OVERHEAD) {
            failWorker("Worker " + i + " exceeded the worker output line limit");
            continue;
          }
        }
        messageHandler(message);
      } else if (line) {
        const visible = filterWorkerStdoutLine(line);
        if (visible) {
          recentStdout = appendRecentText(recentStdout, visible + "\n");
          process.stdout.write(visible + "\n");
        }
      }
    }
    const pendingLimit = output.startsWith(WORKER_MESSAGE_PREFIX) ?
      MAX_WORKER_PROOF_LINE : MAX_WORKER_MESSAGE_LINE;
    if (output.length > pendingLimit) {
      failWorker("Worker " + i + " exceeded the worker output line limit");
      output = "";
    }
  });
  thread.stdout.on("end", function() {
    // Never retain even a truncated IPC payload in unexpected-exit diagnostics.
    if (output && !output.startsWith(WORKER_MESSAGE_PREFIX) &&
        !WORKER_MESSAGE_PREFIX.startsWith(output)) {
      const visible = filterWorkerStdoutLine(output);
      if (visible) {recentStdout = appendRecentText(recentStdout, visible);}
    }
  });
  thread.stderr.setEncoding("utf8");
  thread.stderr.on("data", function(chunk) {
    const filtered = filterWorkerStderr(pendingStderr, chunk);
    pendingStderr = filtered.pending;
    if (!filtered.visible) {return;}
    recentStderr = appendRecentText(recentStderr, filtered.visible);
    process.stderr.write(filtered.visible);
  });
  thread.stderr.on("end", function() {
    const filtered = filterWorkerStderr(pendingStderr, "", true);
    pendingStderr = filtered.pending;
    if (!filtered.visible) {return;}
    recentStderr = appendRecentText(recentStderr, filtered.visible);
    process.stderr.write(filtered.visible);
  });
  thread.stdin.on("error", function(error) {
    failWorker("Worker " + i + " IPC error: " + error.message);
  });
  thread.on("error", function(error) {
    failWorker("Worker " + i + " failed to start: " + error.message);
  });
  thread.on("exit", function(code, signal) {
    const current = workerProcs[i] === thread;
    if (current) {
      delete workerProcs[i];
      workerIds = workerIds.filter((workerId) => workerId !== i);
    }
    if (handleExpectedWorkerExit(thread, i, code, signal,
      workerExitDetail(recentStdout, recentStderr))) {return;}
    if (!current) {return;}
    workerError(messageHandler, i, workerExitMessage(i, code, signal,
      workerExitDetail(recentStdout, recentStderr)
    ));
  });
  workerIds.push(i);
  workerProcs[i] = thread;
}

/** @param {string} recentStdout @param {string} recentStderr */
function workerExitDetail(recentStdout, recentStderr) {
  /** @type {string[]} */
  const detail = [];
  if (recentStdout.trim()) {detail.push("stdout: " + recentStdout.trim());}
  if (recentStderr.trim()) {detail.push("stderr: " + recentStderr.trim());}
  return detail;
}

// Fork worker i through cluster IPC. The first invalid event retires the worker and reports one
// fatal error; later events from it are ignored.
/** @param {number} i @param {NodeJS.ProcessEnv} env @param {WorkerMessageHandler} messageHandler */
function createClusterThread(i, env, messageHandler) {
  const thread = cluster.fork(env);
  thread.on("message", function(message) {
    if (expectedWorkerCloses.get(thread) === "reported") {return;}
    if (isWorkerEvent(message) && message.thread_id === i) {
      messageHandler(message);
      return;
    }
    expectedWorkerCloses.set(thread, "reported");
    workerError(messageHandler, i, "Worker " + i + " emitted invalid worker output");
    thread.kill();
  });
  thread.on("error", function(error) {
    if (expectedWorkerCloses.has(thread)) {return;}
    workerError(messageHandler, i, "Worker " + i + " IPC error: " + error.message);
  });
  thread.on("exit", function(code, signal) {
    workerIds = workerIds.filter((workerId) => workerId !== thread.id);
    if (handleExpectedWorkerExit(thread, i, code, signal)) {return;}
    workerError(messageHandler, i, workerExitMessage(i, code, signal));
  });
  workerIds.push(thread.id);
}

// Map 0..N-1 thread IDs into worker.id (which might not be sequential). Algorithm changes start at
// thread zero because workers own algorithm-specific memory allocations.
/**
 * @param {string} dev
 * @param {WorkerMessageHandler} messageHandler
 * @param {NodeJS.ProcessEnv | ((dev: string, index: number) => NodeJS.ProcessEnv)} [extraEnv]
 */
module.exports.recreate_threads = function(dev, messageHandler, extraEnv = {}) {
  // Reject invalid replacement configuration before retiring the active worker generation.
  /** @type {NodeJS.ProcessEnv[]} */
  const environments = [];
  const threadCount = this.get_dev_threads(dev);
  for (let i = 0; i < threadCount; ++i) {
    const selectedDev = this.get_thread_dev(i, dev);
    if (!selectedDev) {throw new Error(`No device configured for worker ${i}`);}
    const selectedEnv = typeof extraEnv === "function" ? extraEnv(selectedDev, i) : extraEnv;
    const env = childEnv({thread_id: String(i), log_level: String(logLevel()), ...selectedEnv});
    environments.push(env);
  }
  const generation = ++workerGeneration;
  module.exports.closeWorkers(WORKER_REPLACE_CLOSE_GRACE_MS);
  workerIds = [];
  workerProcs = {};
  const currentMessageHandler = forWorkerGeneration(messageHandler, generation);
  for (const [i, env] of environments.entries()) {
    if (useSubprocessWorkers) {
      createSubprocessThread(i, env, currentMessageHandler);
    } else {
      createClusterThread(i, env, currentMessageHandler);
    }
  }
};

// Run step again each time it calls next; callers recurse through asynchronous callbacks.
/** @param {(next: () => void) => unknown} step */
module.exports.repeat = function(step) {
  step(() => module.exports.repeat(step));
};

// The hash and target arithmetic is part of helper's API for pool, miner, and test callers.
const hash = require("./helper/hash");
module.exports.decimalTargetToHex = hash.decimalTargetToHex;
module.exports.diff2target = hash.diff2target;
module.exports.edge_hex2arr = hash.edge_hex2arr;
module.exports.ethDiff2Target = hash.ethDiff2Target;
module.exports.formatHashCount = hash.formatHashCount;
module.exports.formatHashrate = hash.formatHashrate;
module.exports.fullDiff2Target = hash.fullDiff2Target;
module.exports.kawpowTarget2diff = hash.kawpowTarget2diff;
module.exports.pack_msr = hash.pack_msr;
module.exports.pearlhashTarget = hash.pearlhashTarget;
module.exports.pearlhashTargetWork = hash.pearlhashTargetWork;
module.exports.target2diff = hash.target2diff;
module.exports.target256ToWork = hash.target256ToWork;
module.exports.unpack_msr = hash.unpack_msr;
