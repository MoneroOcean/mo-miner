// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

"use strict";

const path    = require("path");
const events  = require("events");
const cluster = require("cluster");
const fs      = require('fs');
const childProcess = require("child_process");

const is_windows_process = process.platform === "win32";
const is_explicit_worker = process.env.MOMINER_CLUSTER_WORKER === "1";
const is_worker_process = is_explicit_worker ||
  (!is_windows_process && !cluster.isMaster);
const use_subprocess_workers = is_windows_process ||
  process.env.MOMINER_USE_SUBPROCESS_WORKERS === "1";
const thread_id = is_worker_process ? Number.parseInt(process.env["thread_id"], 10) : "master";
let worker_ids = []; // active worker ids (cluster.workers can contain not yet closed workers)
let worker_procs = {};
let core_module_for_exit = null;
const worker_message_prefix = "MOMINER_WORKER_MESSAGE ";

function reallyExit(code) {
  setImmediate(() => {
    if (module.exports.exit_now) module.exports.exit_now(code);
    process.exit(code);
  });
}

function childEnv(extra) {
  const env = { ...process.env, ...extra };
  if (process.platform !== "win32") return env;
  return withWindowsWorkerPath(env);
}

function normalizeWindowsPathKey(env) {
  const pathKey = Object.keys(env).find((key) => key.toLowerCase() === "path") || "Path";
  for (const key of Object.keys(env)) {
    if (isDuplicatePathKey(key, pathKey)) delete env[key];
  }
  return pathKey;
}

function isDuplicatePathKey(key, pathKey) {
  return key.toLowerCase() === "path" && key !== pathKey;
}

function withWindowsWorkerPath(env) {
  const appDir = path.dirname(process.execPath);
  return module.exports.withWindowsPathEntries(env, [
    appDir,
    path.join(appDir, "mo-miner"),
    process.cwd(),
    path.join(process.cwd(), "mo-miner"),
    path.join(__dirname, "build", "Release"),
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
  if (process.env.MOMINER_DEBUG_STARTUP) console.error("MOMINER_DEBUG_STARTUP " + str);
}

function appendRecentText(current, chunk, limit = 8192) {
  const next = current + chunk.toString("utf8");
  return next.length > limit ? next.slice(next.length - limit) : next;
}

function log_str(str) {
  return (new Date().toISOString().replace(/T/, ' ').replace(/\..+/, '')) + " " + str;
}

module.exports.log = function(str) {
  console.log(log_str(global.opt.log_level >= 1 ? "[0] " + str : str));
};

module.exports.log1 = function(str) {
  if (global.opt.log_level >= 1) console.log(log_str("[1] " + str));
};

module.exports.log2 = function(str) {
  if (global.opt.log_level >= 2) console.log(log_str("[2] " + str));
};

module.exports.log3 = function(str) {
  if (global.opt.log_level >= 3) console.log(log_str("[3] " + str));
};

module.exports.log_err = function(str) {
  console.error(log_str("ERROR: " + str));
};

module.exports.create_core = function() {
  this.log3("Starting compute core in " + thread_id + " thread");
  const appDir = path.dirname(process.execPath);
  const core_path = firstExistingPath([
    path.join(appDir, "libs", "mo-miner.node"),
    path.join(appDir, "mo-miner.node"),
    path.join(appDir, "mo-miner", "mo-miner.node"),
    path.join(appDir, "build", "Release", "mo-miner.node"),
    path.join(process.cwd(), "libs", "mo-miner.node"),
    path.join(process.cwd(), "mo-miner.node"),
    path.join(process.cwd(), "mo-miner", "mo-miner.node"),
    path.join(__dirname, "libs", "mo-miner.node"),
    path.join(__dirname, "mo-miner.node"),
    path.join(__dirname, "build", "Release", "mo-miner.node"),
  ]);
  debugStartup("requiring " + core_path);
  const core_module = require(core_path);
  debugStartup("required native module");
  core_module_for_exit = core_module;
  let emitter = new events();
  debugStartup("constructing AsyncWorker");
  let worker = new core_module.AsyncWorker(
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
      for (const [key, value] of Object.entries(data || {})) payload[key] = corePayloadValue(value);
      debugStartup("sending " + name + " to native module");
      worker.sendToCpp(name, payload);
      debugStartup("sent " + name + " to native module");
    }
  };
};

function corePayloadValue(value) {
  return value === undefined || value === null ? "" : String(value);
}

module.exports.exit_now = function(code) {
  if (core_module_for_exit && core_module_for_exit.exitNow) {
    core_module_for_exit.exitNow(code);
  }
  process.exit(code);
};

function sendWorkerMessage(type, value) {
  const msg = {type: type, value: value, thread_id: thread_id};
  if (process.send) return process.send(msg);
  process.stdout.write(worker_message_prefix + JSON.stringify(msg) + "\n");
}

function forwardCoreMessages(compute_core) {
  compute_core.from.on("test",        function(v) { sendWorkerMessage("test", v); });
  compute_core.from.on("last_nonce",  function(v) { sendWorkerMessage("last_nonce", v); });
  compute_core.from.on("result",      function(v) { sendWorkerMessage("result", v); });
  compute_core.from.on("hashrate",    function(v) { sendWorkerMessage("hashrate", v); });
  compute_core.from.on("algo_params", function(v) { sendWorkerMessage("algo_params", v); });
  compute_core.from.on("error",       function(v) { sendWorkerMessage("error", v); });
}

function closeWorkerProcess(compute_core, is_exiting_ref) {
  if (is_exiting_ref.value) return reallyExit(0);
  is_exiting_ref.value = true;
  compute_core.emit_to("close");
  setTimeout(function() { reallyExit(0); }, 3000).unref();
}

function installWorkerExitHandlers(close_worker_process) {
  process.on("SIGINT", close_worker_process);
  process.on("SIGTERM", close_worker_process);
  if (process.platform === "win32") process.on("SIGBREAK", close_worker_process);
  else process.on("SIGHUP", close_worker_process);
}

function startWorkerJob(compute_core, msg) {
  // find dev for this specific thread from msg.job.dev list
  msg.job.dev = module.exports.get_thread_dev(thread_id, msg.job.dev);
  msg.job.thread_id = thread_id;
  compute_core.emit_to(msg.type, msg.job);
}

function handleWorkerMessage(compute_core, msg) {
  const handler = workerMessageHandlers[msg.type];
  if (handler) return handler(compute_core, msg);
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
      if (line) handle_msg(JSON.parse(line));
    }
  });
}

module.exports.cluster_process = function() {
  if (!is_worker_process) return false;

  // process worker thread env vars
  global.opt = { log_level: Number.parseInt(process.env["log_level"], 10) };

  let compute_core = this.create_core();

  // send message from worker thread to master thread
  forwardCoreMessages(compute_core);
  compute_core.from.on("close",       function()  {
    process.exitCode = 0;
    if (process.disconnect) process.disconnect();
    reallyExit(0);
  });

  const is_exiting = { value: false };
  const close_worker_process = function() { closeWorkerProcess(compute_core, is_exiting); };
  installWorkerExitHandlers(close_worker_process);
  const handle_msg = function(msg) { handleWorkerMessage(compute_core, msg); };

  // process messages from the master thread
  process.on("message", handle_msg);
  if (!process.send) readStdinMessages(handle_msg);

  return true;
};

// get thread dev stripping ^thread specification from it
function parseThreadDev(dev_part) {
  const m = dev_part.match(/^([^^]+)(?:\^(\d+))?$/);
  return { dev: m ? m[1] : dev_part, threads: m && m[2] ? Number.parseInt(m[2], 10) : 1 };
}

module.exports.is_valid_dev = function(dev) {
  return typeof dev === "string" && dev.split(",").every(function(dev_part) {
    return /^(?:cpu\d*|gpu\d+[oz]?)(?:\*[1-9]\d*)?(?:\^[1-9]\d*)?$/.test(dev_part);
  });
};

module.exports.get_thread_dev = function(thread_id, devs) {
  let thread_count = 0;
  for (const dev_part of devs.split(",")) {
    const parsed = parseThreadDev(dev_part);
    thread_count += parsed.threads;
    if (thread_id < thread_count) return parsed.dev;
  }
  this.log_err("Can't find " + thread_id + " thread device in " + devs + " specification");
  return null;
};

// return number of ^threads in dev specification
module.exports.get_dev_threads = function(dev) {
  let thread_count = 0;
  for (const dev_part of dev.split(",")) thread_count += parseThreadDev(dev_part).threads;
  return thread_count;
};

// return dev *batch value
module.exports.get_dev_batch = function(dev) {
  const m = dev.match(/\*(\d+)$/);
  return m ? Number.parseInt(m[1], 10) : 1;
};

function isCloseMessage(msg) {
  return msg.type === "close";
}

function markExpectedClose(worker, msg) {
  if (isCloseMessage(msg)) worker.expectedClose = true;
}

function isUnexpectedSendError(worker, msg) {
  return !isCloseMessage(msg) && !worker.expectedClose;
}

function sendSubprocessWorker(worker_id, worker, msg) {
  if (!worker || !worker.stdin || !worker.stdin.writable) return null;
  markExpectedClose(worker, msg);
  worker.stdin.write(JSON.stringify(msg) + "\n");
  return { type: "subprocess", id: worker_id, worker };
}

function emitClusterSendError(cluster_worker, msg, error) {
  if (isUnexpectedSendError(cluster_worker, msg)) cluster_worker.emit("error", error);
}

function canSendClusterWorker(cluster_worker) {
  return !cluster_worker.isConnected || cluster_worker.isConnected();
}

function sendClusterMessage(cluster_worker, msg) {
  try {
    cluster_worker.send(msg, function(error) {
      if (error) emitClusterSendError(cluster_worker, msg, error);
    });
  } catch (error) {
    emitClusterSendError(cluster_worker, msg, error);
    return false;
  }
  return true;
}

function sendClusterWorker(worker_id, cluster_worker, msg) {
  if (!cluster_worker) return null;
  markExpectedClose(cluster_worker, msg);
  if (!canSendClusterWorker(cluster_worker)) return null;
  if (!sendClusterMessage(cluster_worker, msg)) return null;
  return { type: "cluster", id: worker_id, worker: cluster_worker };
}

module.exports.messageWorkers = function(msg) {
  const targets = [];
  for (const worker_id of worker_ids) {
    const target = sendSubprocessWorker(worker_id, worker_procs[worker_id], msg) ||
                   sendClusterWorker(worker_id, cluster.workers[worker_id], msg);
    if (target) targets.push(target);
  }
  return targets;
};

function forceCloseSubprocess(worker) {
  if (isSubprocessClosed(worker)) return;
  module.exports.killProcessTree(worker);
}

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

function forceCloseClusterWorker(worker) {
  if (worker.isDead && worker.isDead()) return;
  worker.kill("SIGKILL");
}

function forceCloseWorker(target) {
  const worker = target.worker;
  if (!worker) return;
  if (target.type === "subprocess") return forceCloseSubprocess(worker);
  forceCloseClusterWorker(worker);
}

module.exports.closeWorkers = function(forceAfterMs) {
  const targets = module.exports.messageWorkers({type: "close"});
  if (forceAfterMs !== undefined && forceAfterMs !== null) {
    setTimeout(function() {
      for (const target of targets) forceCloseWorker(target);
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
    env: {...env, MOMINER_CLUSTER_WORKER: "1"},
    stdio: ["pipe", "pipe", "pipe"],
  });
  let output = "";
  let recentStdout = "";
  let recentStderr = "";
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
        process.stdout.write(line + "\n");
      }
    }
  });
  thread.stderr.on("data", function(chunk) {
    recentStderr = appendRecentText(recentStderr, chunk);
    process.stderr.write(chunk);
  });
  thread.on("error", function(error) {
    workerError(messageHandler, i, "Worker " + i + " failed to start: " + error.message);
  });
  thread.on("exit", function(code, signal) {
    if (worker_procs[i] !== thread) return;
    delete worker_procs[i];
    worker_ids = worker_ids.filter((worker_id) => worker_id !== i);
    if (thread.expectedClose) return;
    workerError(messageHandler, i, workerExitMessage(i, code, signal,
      workerExitDetail(recentStdout, recentStderr)
    ));
  });
  worker_ids.push(i);
  worker_procs[i] = thread;
}

function workerExitDetail(recentStdout, recentStderr) {
  const detail = [];
  if (recentStdout.trim()) detail.push("stdout: " + recentStdout.trim());
  if (recentStderr.trim()) detail.push("stderr: " + recentStderr.trim());
  return detail;
}

function createClusterThread(i, env, messageHandler) {
  const thread = cluster.fork(env);
  thread.on("message", messageHandler);
  thread.on("error", function(error) {
    if (thread.expectedClose) return;
    workerError(messageHandler, i, "Worker " + i + " IPC error: " + error.message);
  });
  thread.on("exit", function(code, signal) {
    worker_ids = worker_ids.filter((worker_id) => worker_id !== thread.id);
    if (thread.expectedClose) return;
    workerError(messageHandler, i, workerExitMessage(i, code, signal));
  });
  worker_ids.push(thread.id);
}

// map 0..N-1 thread IDs into worker.id (that might be not sequential)
// need to recreate threads from 0 for every algo change since huge memory reallocations
// can have issues
module.exports.recreate_threads = function(dev, messageHandler) {
  module.exports.closeWorkers(5000);
  //for (const thread of Object.values(thread_id_map)) cluster.workers[thread].kill();
  worker_ids = [];
  worker_procs = {};
  const curr_thread_count = this.get_dev_threads(dev);
  for (let i = 0; i < curr_thread_count; ++ i) {
    const env = childEnv({thread_id: i, log_level: global.opt.log_level});
    if (use_subprocess_workers) createSubprocessThread(i, env, messageHandler);
    else createClusterThread(i, env, messageHandler);
  }
};

// repeat while cb_next is called returns true result with ms delay
module.exports.repeat = function(cb_next, delay) {
  cb_next(function() {
    if (delay) setTimeout(module.exports.repeat, delay, cb_next, delay);
    else return module.exports.repeat(cb_next, delay);
  });
};

// pack opt.default_msrs to it can be more easily passed into compute core
module.exports.pack_msr = function(default_msr) {
  let packed = {};
  for (const [key, val] of Object.entries(default_msr)) {
    packed["msr:" + key] = val.value + "," + val.mask;
  }
  return packed;
};

module.exports.unpack_msr = function(default_msr) {
  let unpacked = {};
  for (const [key, val] of Object.entries(default_msr)) {
    if (!key.startsWith("msr:0x")) continue;
    const parts = val.split(",");
    if (parts.length !== 2) continue;
    unpacked[key.substring(4)] = { value: parts[0], mask: parts[1] };
  }
  return unpacked;
};

module.exports.target2diff = function(target) {
  if (target.length === 8) target = "00000000" + target;
  // need BE -> LE conversion here
  const div = BigInt("0x" + target.match(/.{2}/g).reverse().join(""));
  if (div === BigInt(0)) return 0;
  return BigInt("0xFFFFFFFFFFFFFFFF") / div;
};

module.exports.kawpowTarget2diff = function(target) {
  const target64 = target.slice(0, 16).padEnd(16, "0");
  const div = BigInt("0x" + target64);
  if (div === BigInt(0)) return 0;
  return BigInt("0xFFFFFFFFFFFFFFFF") / div;
};

module.exports.diff2target = function(diff) {
  const diff2 = BigInt(diff);
  if (diff2 <= 0n) return "0000000000000000";
  // Reverse of: div = BigInt(0xFFFFFFFFFFFFFFFF) / diff
  let div = BigInt("0xFFFFFFFFFFFFFFFF") / diff2;
  // Convert to hex without `0x`
  let hexLE = div.toString(16).padStart(16, "0");
  // Reverse LE -> BE
  let hexBE = hexLE.match(/.{2}/g).reverse().join("");
  // If 8-byte (16 hex chars), remove leading zeros to match original style
  if (hexBE.startsWith("00000000")) {
    hexBE = hexBE.slice(8);
  }
  return hexBE;
};

module.exports.edge_hex2arr = function(hex) {
  const pow = [];
  for (let i = 0; i < hex.length; i += 8) pow.push(Number.parseInt(hex.slice(i, i + 8), 16));
  return pow;
}
