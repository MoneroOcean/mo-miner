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

const is_windows_process = process.platform === "win32";
const development_build_platform = is_windows_process ? "win" : "lin";
const is_explicit_worker = process.env["MOM_CLUSTER_WORKER"] === "1";
const is_worker_process = is_explicit_worker ||
  (!is_windows_process && !cluster.isPrimary);
const use_subprocess_workers = is_windows_process ||
  process.env["MOM_USE_SUBPROCESS_WORKERS"] === "1" ||
  (process.env["MOM_GPU_BACKEND"] || "").toLowerCase() === "amd";

/** @returns {number | "master"} */
function processThreadId() {
  if (!is_worker_process) {return "master";}
  const value = process.env["thread_id"] || "";
  const id = Number(value);
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(id)) {
    throw new Error(`Invalid worker thread id: ${value || "<missing>"}`);
  }
  return id;
}

const thread_id = processThreadId();
/** @type {number[]} */
let worker_ids = []; // active worker ids (cluster.workers can contain not yet closed workers)
/** @type {Record<number, SubprocessWorker>} */
let worker_procs = {};
let worker_generation = 0;
/** @type {WeakMap<SubprocessWorker | ClusterWorker, "requested" | "forced" | "reported">} */
const expectedWorkerCloses = new WeakMap();
/** @type {NativeCoreModule | null} */
let core_module_for_exit = null;
let worker_log_level = 0;
const worker_message_prefix = "MOM_WORKER_MESSAGE ";
const max_worker_message_line = 1024 * 1024;
const {MAX_PEARL_PROOF_BASE64, MAX_PROOF_EVENT_OVERHEAD} = require("./helper/worker-protocol");
const max_worker_proof_line = MAX_PEARL_PROOF_BASE64 + MAX_PROOF_EVENT_OVERHEAD;
const diagnostics = require("./helper/diagnostics");
const {filterWorkerStderr, filterWorkerStdoutLine} = diagnostics;
module.exports.filterWorkerStderr = filterWorkerStderr;
module.exports.filterWorkerStdoutLine = filterWorkerStdoutLine;

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

/** @param {unknown} value @returns {value is UnknownRecord} */
function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** @param {unknown} value @returns {value is NativeCoreModule} */
function isNativeCoreModule(value) {
  return isObject(value) && typeof value["AsyncWorker"] === "function" &&
    (value["exitNow"] === undefined || typeof value["exitNow"] === "function");
}

/** @param {unknown} value @returns {value is NativeCoreWorker} */
function isNativeCoreWorker(value) {
  return isObject(value) && typeof value["sendToCpp"] === "function";
}

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
  const nonce = value["nonce"];
  if (nonce !== undefined &&
      !(typeof nonce === "string" && /^[0-9a-f]{1,16}$/i.test(nonce)) &&
      !(typeof nonce === "number" && Number.isSafeInteger(nonce) && nonce >= 0)) {
    return false;
  }
  const mask = value["nicehash_mask"];
  if (mask !== undefined && (typeof mask !== "string" || !/^[0-9a-f]{1,16}$/i.test(mask))) {
    return false;
  }
  for (const field of ["job_id", "pool_id", "worker_id"]) {
    const fieldValue = value[field];
    if (fieldValue !== undefined && typeof fieldValue !== "string" &&
        (typeof fieldValue !== "number" || !Number.isSafeInteger(fieldValue))) {
      return false;
    }
  }
  return true;
}

/** @param {object} job @param {string} key */
function hasJobIdentifier(job, key) {
  const value = Reflect.get(job, key);
  return typeof value === "string" ||
    typeof value === "number" && Number.isSafeInteger(value);
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
  return typeof job.target === "string" && /^[0-9a-f]{1,64}$/i.test(job.target.replace(/^0x/i, "")) &&
    hasJobIdentifier(job, "job_id") && hasJobIdentifier(job, "pool_id") &&
    hasJobIdentifier(job, "worker_id") &&
    typeof job.job_token === "string" && job.job_token.length > 0;
}

/** @param {unknown} value @returns {value is WorkerEvent} */
function isWorkerEvent(value) {
  return Boolean(isObject(value) && isObject(value["value"]) && typeof value["type"] === "string" &&
    typeof value["thread_id"] === "number" && Number.isSafeInteger(value["thread_id"]) &&
    value["thread_id"] >= 0);
}
module.exports.is_worker_event = isWorkerEvent;

/** @param {number} code */
function reallyExit(code) {
  setImmediate(() => module.exports.exit_now(code));
}

/** @param {NodeJS.ProcessEnv} extra */
function childEnv(extra) {
  const env = {...process.env, ...extra};
  return process.platform === "win32" ? withWindowsWorkerPath(env) : env;
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
    path.join(__dirname, "build", development_build_platform, "Release"),
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
function log_str(str) {
  return (new Date().toISOString().replace(/T/, " ").replace(/\..+/, "")) + " " + str;
}

function logLevel() {
  return is_worker_process ? worker_log_level : global.opt.log_level;
}

/** @param {string} str */
module.exports.log = function(str) {
  console.log(log_str(logLevel() >= 1 ? "[0] " + str : str));
};

/** @param {number} level */
function makeLevelLogger(level) {
  return function(/** @type {string} */ str) {
    if (logLevel() >= level) {console.log(log_str("[" + level + "] " + str));}
  };
}

const compute_message_redacted_keys = new Set(["plain_proof", "worker_id", "login", "pass", "wallet"]);

/** @param {unknown} value @returns {string} */
function format_compute_message(value) {
  try {
    const serialized = JSON.stringify(value, (key, nested) =>
      compute_message_redacted_keys.has(key) ? "<redacted>" : nested);
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
  console.error(log_str("ERROR: " + str));
};

/** @returns {ComputeCore} */
module.exports.create_core = function() {
  this.log3("Starting compute core in " + thread_id + " thread");
  const appDir = path.dirname(process.execPath);
  // The selected addon and runtime are one choice; a missing addon must not load a different worker.
  const core_path = process.env["MOM_NATIVE_PATH"] || firstExistingPath([
    path.join(appDir, "libs", "mom.node"),
    path.join(appDir, "mom.node"),
    path.join(appDir, "mom", "mom.node"),
    path.join(appDir, "build", development_build_platform, "Release", "mom.node"),
    path.join(process.cwd(), "libs", "mom.node"),
    path.join(process.cwd(), "mom.node"),
    path.join(process.cwd(), "mom", "mom.node"),
    path.join(__dirname, "libs", "mom.node"),
    path.join(__dirname, "mom.node"),
    path.join(__dirname, "build", development_build_platform, "Release", "mom.node"),
  ]);
  debugStartup("requiring " + core_path);
  /** @type {unknown} */
  const loaded_module = require(core_path);
  if (!isNativeCoreModule(loaded_module)) {throw new Error("Invalid native compute-core module");}
  const core_module = loaded_module;
  debugStartup("required native module");
  core_module_for_exit = core_module;
  const emitter = new EventEmitter();
  debugStartup("constructing AsyncWorker");
  const worker = new core_module.AsyncWorker(
    function(/** @type {string} */ name, /** @type {unknown} */ value) {
      module.exports.log3("Getting from compute core " + thread_id + " " + name + " message: " +
                          format_compute_message(value));
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
      module.exports.log3("Sending to compute core " + thread_id + " " + name + " message: " +
                          format_compute_message(data));
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
  if (core_module_for_exit && core_module_for_exit.exitNow) {
    core_module_for_exit.exitNow(code);
  }
  process.exit(code);
};

/** @param {string} type @param {unknown} value */
function sendWorkerMessage(type, value) {
  const msg = {type, value, thread_id};
  if (process.send) {return process.send(msg);}
  return process.stdout.write(worker_message_prefix + JSON.stringify(msg) + "\n");
}

/** @param {ComputeCore} compute_core */
function forwardCoreMessages(compute_core) {
  for (const name of ["test", "last_nonce", "result", "hashrate", "algo_params"]) {
    compute_core.from.on(name, function(v) { sendWorkerMessage(name, v); });
  }
  compute_core.from.on("error", function(error) {
    const message = error instanceof Error ? error.message :
      isObject(error) && typeof error["message"] === "string" ? error["message"] : String(error);
    sendWorkerMessage("error", {message});
  });
}

/** @param {() => void} close_worker_process */
function installWorkerExitHandlers(close_worker_process) {
  let receivedSignal = false;
  const receiveSignal = function() {
    if (receivedSignal) {return reallyExit(0);}
    receivedSignal = true;
    close_worker_process();
  };
  process.on("SIGINT", receiveSignal);
  process.on("SIGTERM", receiveSignal);
  if (process.platform === "win32") {
    process.on("SIGBREAK", receiveSignal);
  } else {
    process.on("SIGHUP", receiveSignal);
  }
}

/** @param {ComputeCore} compute_core @param {WorkerJobCommand} msg @param {number} workerThreadId */
function startWorkerJob(compute_core, msg, workerThreadId) {
  // Worker topology is owned at dispatch; capture the count before per-worker device selection/tuning.
  const workerThreadCount = module.exports.get_dev_threads(msg.job.dev);
  // find dev for this specific thread from msg.job.dev list
  const selected = gpuTuning.parseDeviceEntry(
    module.exports.get_thread_dev(workerThreadId, msg.job.dev), msg.job.algo);
  gpuTuning.applyNativeJobTuning(msg.job, selected, msg.job.algo);
  msg.job.thread_id = workerThreadId;
  msg.job.thread_num = workerThreadCount;
  compute_core.emit_to(msg.type, msg.job);
}

/** @param {ComputeCore} compute_core @param {WorkerCommand} msg @param {number} workerThreadId */
function handleWorkerMessage(compute_core, msg, workerThreadId) {
  if (msg.type === "job" || msg.type === "bench" || msg.type === "test") {
    return startWorkerJob(compute_core, msg, workerThreadId);
  }
  return compute_core.emit_to(msg.type);
}

/** @param {(message: unknown) => boolean} handle_msg @param {() => void} on_end */
function readStdinMessages(handle_msg, on_end) {
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
      if (line.length > max_worker_message_line) {
        return fail("Worker IPC input exceeded the message line limit");
      }
      if (!line) {continue;}
      let message;
      try {
        message = JSON.parse(line);
      } catch {
        return fail("Worker IPC input was malformed");
      }
      if (!handle_msg(message)) {
        return fail("Worker IPC input was malformed");
      }
    }
    if (input.length > max_worker_message_line) {
      return fail("Worker IPC input exceeded the message line limit");
    }
  });
  process.stdin.once("end", function() {
    if (failed) {return;}
    if (input.trim()) {return fail("Worker IPC input was malformed");}
    on_end();
  });
}

module.exports.cluster_process = function() {
  if (!is_worker_process) {return false;}
  if (typeof thread_id !== "number") {throw new Error("Worker process has no numeric thread id");}
  const workerThreadId = thread_id;

  // process worker thread env vars
  worker_log_level = Number.parseInt(process.env["log_level"] || "0", 10);

  const compute_core = this.create_core();
  let is_exiting = false;
  const close_worker_process = function() {
    if (is_exiting) {return;}
    is_exiting = true;
    compute_core.emit_to("close");
  };

  // send message from worker thread to master thread
  forwardCoreMessages(compute_core);
  compute_core.from.on("close",       function()  {
    is_exiting = true;
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

  installWorkerExitHandlers(close_worker_process);
  const handle_msg = function(/** @type {unknown} */ msg) {
    if (!isWorkerCommand(msg)) {return false;}
    if (msg.type === "close") {
      close_worker_process();
      return true;
    }
    if (is_exiting) {return true;}
    handleWorkerMessage(compute_core, msg, workerThreadId);
    return true;
  };

  // process messages from the master thread
  process.on("message", function(msg) {
    if (handle_msg(msg)) {return;}
    module.exports.log_err("Worker IPC input was malformed");
    reallyExit(1);
  });
  if (!process.send) {readStdinMessages(handle_msg, close_worker_process);}

  return true;
};

// get thread dev stripping ^thread specification from it
/** @param {string} dev_part */
function parseThreadDev(dev_part) {
  const parsed = gpuTuning.parseDeviceEntry(dev_part);
  const processSuffix = parsed.processes > 1 ? `^${parsed.processes}` : "";
  return {
    // Keep *B intact here: its primary-field meaning is algorithm-specific and
    // is resolved in the worker, where the job's algorithm is available.
    dev: dev_part.slice(0, dev_part.length - processSuffix.length),
    threads: parsed.processes,
  };
}

/** @param {unknown} dev @param {string} [algo] */
module.exports.is_valid_dev = function(dev, algo = "") {
  try {
    gpuTuning.parseDeviceList(dev, algo);
    return true;
  } catch {
    return false;
  }
};

/** @param {number} thread_id @param {string} devs */
module.exports.get_thread_dev = function(thread_id, devs) {
  let thread_count = 0;
  for (const dev_part of devs.split(",")) {
    const parsed = parseThreadDev(dev_part);
    thread_count += parsed.threads;
    if (thread_id < thread_count) {return parsed.dev;}
  }
  this.log_err("Can't find " + thread_id + " thread device in " + devs + " specification");
  return null;
};

// return number of ^threads in dev specification
/** @param {string} dev */
module.exports.get_dev_threads = function(dev) {
  let thread_count = 0;
  for (const dev_part of dev.split(",")) {thread_count += parseThreadDev(dev_part).threads;}
  return thread_count;
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
 * @param {number} worker_id
 * @param {SubprocessWorker | undefined} worker
 * @param {WorkerCommand} msg
 * @returns {WorkerTarget | null}
 */
function sendSubprocessWorker(worker_id, worker, msg) {
  if (!worker) {return null;}
  markExpectedClose(worker, msg);
  /** @type {WorkerTarget} */
  const target = {type: "subprocess", id: worker_id, worker};
  if (!worker.stdin || !worker.stdin.writable) {return msg.type === "close" ? target : null;}
  worker.stdin.write(JSON.stringify(msg) + "\n");
  return target;
}

/** @param {ClusterWorker} cluster_worker @param {WorkerCommand} msg @param {Error} error */
function emitClusterSendError(cluster_worker, msg, error) {
  if (isUnexpectedSendError(cluster_worker, msg)) {cluster_worker.emit("error", error);}
}

/** @param {ClusterWorker} cluster_worker */
function canSendClusterWorker(cluster_worker) {
  return !cluster_worker.isConnected || cluster_worker.isConnected();
}

/** @param {ClusterWorker} cluster_worker @param {WorkerCommand} msg */
function sendClusterMessage(cluster_worker, msg) {
  try {
    cluster_worker.send(msg, function(error) {
      if (error) {emitClusterSendError(cluster_worker, msg, error);}
    });
  } catch (error) {
    emitClusterSendError(cluster_worker, msg,
      error instanceof Error ? error : new Error(String(error)));
    return false;
  }
  return true;
}

/**
 * @param {number} worker_id
 * @param {ClusterWorker | undefined} cluster_worker
 * @param {WorkerCommand} msg
 * @returns {WorkerTarget | null}
 */
function sendClusterWorker(worker_id, cluster_worker, msg) {
  if (!cluster_worker) {return null;}
  markExpectedClose(cluster_worker, msg);
  /** @type {WorkerTarget} */
  const target = {type: "cluster", id: worker_id, worker: cluster_worker};
  if (!canSendClusterWorker(cluster_worker) || !sendClusterMessage(cluster_worker, msg)) {
    return msg.type === "close" ? target : null;
  }
  return target;
}

/** @param {WorkerCommand} msg @returns {WorkerTarget[]} */
module.exports.messageWorkers = function(msg) {
  if (!isWorkerCommand(msg)) {throw new Error("Invalid worker command");}
  /** @type {WorkerTarget[]} */
  const targets = [];
  for (const worker_id of worker_ids) {
    const target = sendSubprocessWorker(worker_id, worker_procs[worker_id], msg) ||
                   sendClusterWorker(worker_id, cluster.workers?.[worker_id], msg);
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
  if (!is_windows_process || !worker.pid) {
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

/** @param {WorkerMessageHandler} messageHandler @param {number} thread_id @param {string} message */
function workerError(messageHandler, thread_id, message) {
  messageHandler({
    type: "error",
    value: {message, fatal: true},
    thread_id
  });
}

/**
 * @param {number} thread_id
 * @param {number | null} code
 * @param {string | null} signal
 * @param {string[]} [detail]
 */
function workerExitMessage(thread_id, code, signal, detail = []) {
  return "Worker " + thread_id + " exited unexpectedly" +
    (signal ? " with signal " + signal : " with code " + code) +
    (detail.length ? ". " + detail.join(" | ") : "");
}

/**
 * @param {SubprocessWorker | ClusterWorker} worker
 * @param {number} worker_id
 * @param {number | null} code
 * @param {string | null} signal
 * @param {string[]} [detail]
 * @returns {boolean}
 */
function handleExpectedWorkerExit(worker, worker_id, code, signal, detail = []) {
  const reason = expectedWorkerCloses.get(worker);
  if (reason === "requested") {
    if (code !== 0 || signal !== null) {
      module.exports.log_err(workerExitMessage(worker_id, code, signal, detail));
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
    if (generation === worker_generation) {
      messageHandler(message);
    }
  };
}

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
      if (line.startsWith(worker_message_prefix)) {
        if (failureReported) {continue;}
        let message;
        try {
          if (line.length > max_worker_proof_line) {throw new Error("oversized worker output");}
          message = JSON.parse(line.slice(worker_message_prefix.length));
        } catch {
          failWorker("Worker " + i + " emitted malformed worker output");
          output = "";
          break;
        }
        if (!isWorkerEvent(message) || message.thread_id !== i) {
          failWorker("Worker " + i + " emitted invalid worker output");
          continue;
        }
        if (line.length > max_worker_message_line) {
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
    const pendingLimit = output.startsWith(worker_message_prefix) ?
      max_worker_proof_line : max_worker_message_line;
    if (output.length > pendingLimit) {
      failWorker("Worker " + i + " exceeded the worker output line limit");
      output = "";
    }
  });
  thread.stdout.on("end", function() {
    // Never retain even a truncated IPC payload in unexpected-exit diagnostics.
    if (output && !output.startsWith(worker_message_prefix) &&
        !worker_message_prefix.startsWith(output)) {
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
    const current = worker_procs[i] === thread;
    if (current) {
      delete worker_procs[i];
      worker_ids = worker_ids.filter((worker_id) => worker_id !== i);
    }
    if (handleExpectedWorkerExit(thread, i, code, signal,
      workerExitDetail(recentStdout, recentStderr))) {return;}
    if (!current) {return;}
    workerError(messageHandler, i, workerExitMessage(i, code, signal,
      workerExitDetail(recentStdout, recentStderr)
    ));
  });
  worker_ids.push(i);
  worker_procs[i] = thread;
}

/** @param {string} recentStdout @param {string} recentStderr */
function workerExitDetail(recentStdout, recentStderr) {
  /** @type {string[]} */
  const detail = [];
  if (recentStdout.trim()) {detail.push("stdout: " + recentStdout.trim());}
  if (recentStderr.trim()) {detail.push("stderr: " + recentStderr.trim());}
  return detail;
}

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
    worker_ids = worker_ids.filter((worker_id) => worker_id !== thread.id);
    if (handleExpectedWorkerExit(thread, i, code, signal)) {return;}
    workerError(messageHandler, i, workerExitMessage(i, code, signal));
  });
  worker_ids.push(thread.id);
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
  const curr_thread_count = this.get_dev_threads(dev);
  for (let i = 0; i < curr_thread_count; ++i) {
    const selectedDev = this.get_thread_dev(i, dev);
    if (!selectedDev) {throw new Error(`No device configured for worker ${i}`);}
    const selectedEnv = typeof extraEnv === "function" ? extraEnv(selectedDev, i) : extraEnv;
    const env = childEnv({thread_id: String(i), log_level: String(logLevel()), ...selectedEnv});
    environments.push(env);
  }
  const generation = ++worker_generation;
  module.exports.closeWorkers(WORKER_REPLACE_CLOSE_GRACE_MS);
  worker_ids = [];
  worker_procs = {};
  const currentMessageHandler = forWorkerGeneration(messageHandler, generation);
  for (const [i, env] of environments.entries()) {
    if (use_subprocess_workers) {
      createSubprocessThread(i, env, currentMessageHandler);
    } else {
      createClusterThread(i, env, currentMessageHandler);
    }
  }
};

// Re-run cb_next each time it invokes its callback, waiting `delay` ms between
// runs (or immediately/recursively when delay is falsy).
/** @param {(next: () => void) => unknown} cb_next @param {number} [delay] */
module.exports.repeat = function(cb_next, delay) {
  cb_next(function() {
    if (delay) {
      setTimeout(module.exports.repeat, delay, cb_next, delay);
    } else {
      module.exports.repeat(cb_next, delay);
    }
  });
};

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
