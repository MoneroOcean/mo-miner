// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

"use strict";

const path    = require("path");
const events  = require("events");
const cluster = require("cluster");
const fs      = require("fs");
const childProcess = require("child_process");
const gpuTuning = require("./gpu-tuning");

const is_windows_process = process.platform === "win32";
const development_build_platform = is_windows_process ? "win" : "lin";
const is_explicit_worker = process.env.MOM_CLUSTER_WORKER === "1";
const is_worker_process = is_explicit_worker ||
  (!is_windows_process && !cluster.isMaster);
const use_subprocess_workers = is_windows_process ||
  process.env.MOM_USE_SUBPROCESS_WORKERS === "1" ||
  (process.env.MOM_GPU_BACKEND || "").toLowerCase() === "amd";
const thread_id = is_worker_process ? Number.parseInt(process.env.thread_id, 10) : "master";
let worker_ids = []; // active worker ids (cluster.workers can contain not yet closed workers)
let worker_procs = {};
let core_module_for_exit = null;
const worker_message_prefix = "MOM_WORKER_MESSAGE ";
const diagnostics = require("./helper/diagnostics");
const { filterWorkerStderr } = diagnostics;
Object.assign(module.exports, diagnostics);

function reallyExit(code) {
  setImmediate(() => {
    if (module.exports.exit_now) {module.exports.exit_now(code);}
    process.exit(code);
  });
}

function childEnv(extra) {
  const env = { ...process.env, ...extra };
  return process.platform === "win32" ? withWindowsWorkerPath(env) : env;
}

function normalizeWindowsPathKey(env) {
  // Windows env var names are case-insensitive; collapse any stray PATH variants
  // (e.g. "Path" and "PATH") onto a single canonical key.
  const pathKey = Object.keys(env).find((key) => key.toLowerCase() === "path") || "Path";
  for (const key of Object.keys(env)) {
    if (key !== pathKey && key.toLowerCase() === "path") {delete env[key];}
  }
  return pathKey;
}

function withWindowsWorkerPath(env) {
  const appDir = path.dirname(process.execPath);
  return module.exports.withWindowsPathEntries(env, [
    env.MOM_NATIVE_PATH && path.dirname(env.MOM_NATIVE_PATH),
    appDir,
    path.join(appDir, "mom"),
    process.cwd(),
    path.join(process.cwd(), "mom"),
    path.join(__dirname, "build", development_build_platform, "Release"),
  ]);
}

module.exports.withWindowsPathEntries = function(env, entries) {
  const pathKey = normalizeWindowsPathKey(env);
  const pathValue = env[pathKey] || "";
  env[pathKey] = [...entries, pathValue].filter(Boolean).join(path.delimiter);
  return env;
};

function firstExistingPath(paths) {
  return paths.find((filePath) => filePath && fs.existsSync(filePath)) || paths[paths.length - 1];
}

function debugStartup(str) {
  if (process.env.MOM_DEBUG_STARTUP) {console.error("MOM_DEBUG_STARTUP " + str);}
}

function appendRecentText(current, chunk, limit = 8192) {
  const next = current + chunk.toString("utf8");
  return next.length > limit ? next.slice(next.length - limit) : next;
}

function log_str(str) {
  return (new Date().toISOString().replace(/T/, " ").replace(/\..+/, "")) + " " + str;
}

module.exports.log = function(str) {
  console.log(log_str(global.opt.log_level >= 1 ? "[0] " + str : str));
};

function makeLevelLogger(level) {
  return function(str) {
    if (global.opt.log_level >= level) {console.log(log_str("[" + level + "] " + str));}
  };
}

module.exports.log1 = makeLevelLogger(1);
module.exports.log2 = makeLevelLogger(2);
module.exports.log3 = makeLevelLogger(3);

module.exports.log_err = function(str) {
  console.error(log_str("ERROR: " + str));
};

module.exports.create_core = function() {
  this.log3("Starting compute core in " + thread_id + " thread");
  const appDir = path.dirname(process.execPath);
  const core_path = firstExistingPath([
    process.env.MOM_NATIVE_PATH,
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
  const core_module = require(core_path);
  debugStartup("required native module");
  core_module_for_exit = core_module;
  const emitter = new events();
  debugStartup("constructing AsyncWorker");
  const worker = new core_module.AsyncWorker(
    function(name, value) {
      module.exports.log3("Getting from compute core " + thread_id + " " + name + " message: " +
                          JSON.stringify(value));
      emitter.emit(name, value);
    },
    function ()     { emitter.emit("close"); },
    function(error) { emitter.emit("error", error); },
    {} // no extra options
  );
  debugStartup("constructed AsyncWorker");
  return {
    from:    emitter,
    emit_to: function(name, data) {
      module.exports.log3("Sending to compute core " + thread_id + " " + name + " message: " +
                          JSON.stringify(data));
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

module.exports.exit_now = function(code) {
  if (core_module_for_exit && core_module_for_exit.exitNow) {
    core_module_for_exit.exitNow(code);
  }
  process.exit(code);
};

function sendWorkerMessage(type, value) {
  const msg = {type, value, thread_id};
  if (process.send) {return process.send(msg);}
  process.stdout.write(worker_message_prefix + JSON.stringify(msg) + "\n");
}

function forwardCoreMessages(compute_core) {
  for (const name of ["test", "last_nonce", "result", "hashrate", "algo_params", "error"]) {
    compute_core.from.on(name, function(v) { sendWorkerMessage(name, v); });
  }
}

function closeWorkerProcess(compute_core, is_exiting_ref) {
  if (is_exiting_ref.value) {return reallyExit(0);}
  is_exiting_ref.value = true;
  compute_core.emit_to("close");
  setTimeout(function() { reallyExit(0); }, 3000).unref();
}

function installWorkerExitHandlers(close_worker_process) {
  process.on("SIGINT", close_worker_process);
  process.on("SIGTERM", close_worker_process);
  if (process.platform === "win32") {process.on("SIGBREAK", close_worker_process);}
  else {process.on("SIGHUP", close_worker_process);}
}

function startWorkerJob(compute_core, msg) {
  // find dev for this specific thread from msg.job.dev list
  const selected = gpuTuning.parseDeviceEntry(
    module.exports.get_thread_dev(thread_id, msg.job.dev), msg.job.algo);
  gpuTuning.applyNativeJobTuning(msg.job, selected, msg.job.algo);
  msg.job.thread_id = thread_id;
  compute_core.emit_to(msg.type, msg.job);
}

function handleWorkerMessage(compute_core, msg) {
  const handler = workerMessageHandlers[msg.type];
  if (handler) {return handler(compute_core, msg);}
  module.exports.log_err("Unknown thread message");
}

const workerMessageHandlers = {
  job: startWorkerJob,
  bench: startWorkerJob,
  test: startWorkerJob,
  pause: function(compute_core, msg) { compute_core.emit_to(msg.type); },
  close: function(compute_core, msg) { compute_core.emit_to(msg.type); },
};

function readStdinMessages(handle_msg) {
  let input = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", function(chunk) {
    input += chunk;
    let eol;
    while ((eol = input.indexOf("\n")) !== -1) {
      const line = input.slice(0, eol);
      input = input.slice(eol + 1);
      if (line) {handle_msg(JSON.parse(line));}
    }
  });
}

module.exports.cluster_process = function() {
  if (!is_worker_process) {return false;}

  // process worker thread env vars
  global.opt = { log_level: Number.parseInt(process.env.log_level, 10) };

  const compute_core = this.create_core();

  // send message from worker thread to master thread
  forwardCoreMessages(compute_core);
  compute_core.from.on("close",       function()  {
    process.exitCode = 0;
    if (process.disconnect) {process.disconnect();}
    reallyExit(0);
  });

  const is_exiting = { value: false };
  const close_worker_process = function() { closeWorkerProcess(compute_core, is_exiting); };
  installWorkerExitHandlers(close_worker_process);
  const handle_msg = function(msg) { handleWorkerMessage(compute_core, msg); };

  // process messages from the master thread
  process.on("message", handle_msg);
  if (!process.send) {readStdinMessages(handle_msg);}

  return true;
};

// get thread dev stripping ^thread specification from it
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

module.exports.is_valid_dev = function(dev, algo = "") {
  try {
    gpuTuning.parseDeviceList(dev, algo);
    return true;
  } catch {
    return false;
  }
};

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
module.exports.get_dev_threads = function(dev) {
  let thread_count = 0;
  for (const dev_part of dev.split(",")) {thread_count += parseThreadDev(dev_part).threads;}
  return thread_count;
};

// return dev *batch value
module.exports.get_dev_batch = function(dev) {
  try {
    const tuning = gpuTuning.parseDeviceEntry(dev).tuning;
    return tuning.intensity || tuning.m || 1;
  } catch {
    return 1;
  }
};

function markExpectedClose(worker, msg) {
  if (msg.type === "close") {worker.expectedClose = true;}
}

function isUnexpectedSendError(worker, msg) {
  return msg.type !== "close" && !worker.expectedClose;
}

function sendSubprocessWorker(worker_id, worker, msg) {
  if (!worker || !worker.stdin || !worker.stdin.writable) {return null;}
  markExpectedClose(worker, msg);
  worker.stdin.write(JSON.stringify(msg) + "\n");
  return { type: "subprocess", id: worker_id, worker };
}

function emitClusterSendError(cluster_worker, msg, error) {
  if (isUnexpectedSendError(cluster_worker, msg)) {cluster_worker.emit("error", error);}
}

function canSendClusterWorker(cluster_worker) {
  return !cluster_worker.isConnected || cluster_worker.isConnected();
}

function sendClusterMessage(cluster_worker, msg) {
  try {
    cluster_worker.send(msg, function(error) {
      if (error) {emitClusterSendError(cluster_worker, msg, error);}
    });
  } catch (error) {
    emitClusterSendError(cluster_worker, msg, error);
    return false;
  }
  return true;
}

function sendClusterWorker(worker_id, cluster_worker, msg) {
  if (!cluster_worker) {return null;}
  markExpectedClose(cluster_worker, msg);
  if (!canSendClusterWorker(cluster_worker)) {return null;}
  if (!sendClusterMessage(cluster_worker, msg)) {return null;}
  return { type: "cluster", id: worker_id, worker: cluster_worker };
}

module.exports.messageWorkers = function(msg) {
  const targets = [];
  for (const worker_id of worker_ids) {
    const target = sendSubprocessWorker(worker_id, worker_procs[worker_id], msg) ||
                   sendClusterWorker(worker_id, cluster.workers[worker_id], msg);
    if (target) {targets.push(target);}
  }
  return targets;
};

function isSubprocessClosed(worker) {
  return worker.exitCode !== null || worker.signalCode !== null || worker.killed;
}

module.exports.killProcessTree = function(worker, signal = "SIGKILL") {
  if (!is_windows_process || !worker.pid) {
    worker.kill(signal);
    return false;
  }
  const killer = childProcess.spawn("taskkill", ["/pid", String(worker.pid), "/t", "/f"], {
    stdio: "ignore",
  });
  killer.on("error", function() { worker.kill(signal); });
  return true;
};

function forceCloseWorker(target) {
  const worker = target.worker;
  if (!worker) {return;}
  if (target.type === "subprocess") {
    if (!isSubprocessClosed(worker)) {module.exports.killProcessTree(worker);}
  } else if (!worker.isDead || !worker.isDead()) {
    worker.kill("SIGKILL");
  }
}

module.exports.closeWorkers = function(forceAfterMs) {
  const targets = module.exports.messageWorkers({type: "close"});
  if (forceAfterMs != null) {
    setTimeout(function() {
      for (const target of targets) {forceCloseWorker(target);}
    }, forceAfterMs).unref();
  }
  return targets;
};

function workerError(messageHandler, thread_id, message) {
  messageHandler({
    type: "error",
    value: { message },
    thread_id
  });
}

function workerExitMessage(thread_id, code, signal, detail = []) {
  return "Worker " + thread_id + " exited unexpectedly" +
    (signal ? " with signal " + signal : " with code " + code) +
    (detail.length ? ". " + detail.join(" | ") : "");
}

function createSubprocessThread(i, env, messageHandler) {
  const thread = childProcess.spawn(process.execPath, process.argv.slice(1), {
    env: {...env, MOM_CLUSTER_WORKER: "1"},
    stdio: ["pipe", "pipe", "pipe"],
  });
  let output = "";
  let recentStdout = "";
  let recentStderr = "";
  let pendingStderr = "";
  thread.stdout.setEncoding("utf8");
  thread.stdout.on("data", function(chunk) {
    recentStdout = appendRecentText(recentStdout, chunk);
    output += chunk;
    let eol;
    while ((eol = output.indexOf("\n")) !== -1) {
      const line = output.slice(0, eol);
      output = output.slice(eol + 1);
      if (line.startsWith(worker_message_prefix)) {
        messageHandler(JSON.parse(line.slice(worker_message_prefix.length)));
      } else if (line) {
        const visible = module.exports.filterWorkerStdoutLine(line);
        if (visible) {process.stdout.write(visible + "\n");}
      }
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
  thread.on("error", function(error) {
    workerError(messageHandler, i, "Worker " + i + " failed to start: " + error.message);
  });
  thread.on("exit", function(code, signal) {
    if (worker_procs[i] !== thread) {return;}
    delete worker_procs[i];
    worker_ids = worker_ids.filter((worker_id) => worker_id !== i);
    if (thread.expectedClose) {return;}
    workerError(messageHandler, i, workerExitMessage(i, code, signal,
      workerExitDetail(recentStdout, recentStderr)
    ));
  });
  worker_ids.push(i);
  worker_procs[i] = thread;
}

function workerExitDetail(recentStdout, recentStderr) {
  const detail = [];
  if (recentStdout.trim()) {detail.push("stdout: " + recentStdout.trim());}
  if (recentStderr.trim()) {detail.push("stderr: " + recentStderr.trim());}
  return detail;
}

function createClusterThread(i, env, messageHandler) {
  const thread = cluster.fork(env);
  thread.on("message", messageHandler);
  thread.on("error", function(error) {
    if (thread.expectedClose) {return;}
    workerError(messageHandler, i, "Worker " + i + " IPC error: " + error.message);
  });
  thread.on("exit", function(code, signal) {
    worker_ids = worker_ids.filter((worker_id) => worker_id !== thread.id);
    if (thread.expectedClose) {return;}
    workerError(messageHandler, i, workerExitMessage(i, code, signal));
  });
  worker_ids.push(thread.id);
}

// map 0..N-1 thread IDs into worker.id (that might be not sequential)
// need to recreate threads from 0 for every algo change since huge memory reallocations
// can have issues
module.exports.recreate_threads = function(dev, messageHandler, extraEnv = {}) {
  module.exports.closeWorkers(5000);
  worker_ids = [];
  worker_procs = {};
  const curr_thread_count = this.get_dev_threads(dev);
  for (let i = 0; i < curr_thread_count; ++ i) {
    const selectedDev = this.get_thread_dev(i, dev);
    const selectedEnv = typeof extraEnv === "function" ? extraEnv(selectedDev, i) : extraEnv;
    const env = childEnv({thread_id: i, log_level: global.opt.log_level, ...selectedEnv});
    if (use_subprocess_workers) {createSubprocessThread(i, env, messageHandler);}
    else {createClusterThread(i, env, messageHandler);}
  }
};

// Re-run cb_next each time it invokes its callback, waiting `delay` ms between
// runs (or immediately/recursively when delay is falsy).
module.exports.repeat = function(cb_next, delay) {
  cb_next(function() {
    if (delay) {setTimeout(module.exports.repeat, delay, cb_next, delay);}
    else {module.exports.repeat(cb_next, delay);}
  });
};

Object.assign(module.exports, require("./helper/hash"));
