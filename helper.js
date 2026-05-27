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
const thread_id = is_worker_process ? parseInt(process.env["thread_id"]) : "master";
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

  const pathKey = Object.keys(env).find((key) => key.toLowerCase() === "path") || "Path";
  const pathValue = env[pathKey] || "";
  const appDir = path.dirname(process.execPath);
  for (const key of Object.keys(env)) {
    if (key.toLowerCase() === "path" && key !== pathKey) delete env[key];
  }
  env[pathKey] = [
    appDir,
    path.join(appDir, "mo-miner"),
    process.cwd(),
    path.join(process.cwd(), "mo-miner"),
    path.join(__dirname, "build", "Release"),
    pathValue,
  ]
    .filter(Boolean)
    .join(path.delimiter);
  return env;
}

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
      for (const [key, value] of Object.entries(data ? data : {})) {
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

module.exports.cluster_process = function() {
  if (!is_worker_process) return false;

  // process worker thread env vars
  global.opt = { log_level: parseInt(process.env["log_level"]) };

  let compute_core = this.create_core();

  // send message from worker thread to master thread
  function send_msg(type, value) {
    const msg = {type: type, value: value, thread_id: thread_id};
    if (process.send) return process.send(msg);
    process.stdout.write(worker_message_prefix + JSON.stringify(msg) + "\n");
  }
  compute_core.from.on("test",        function(v) { send_msg("test", v); });
  compute_core.from.on("last_nonce",  function(v) { send_msg("last_nonce", v); });
  compute_core.from.on("result",      function(v) { send_msg("result", v); });
  compute_core.from.on("hashrate",    function(v) { send_msg("hashrate", v); });
  compute_core.from.on("algo_params", function(v) { send_msg("algo_params", v); });
  compute_core.from.on("error",       function(v) { send_msg("error", v); });
  compute_core.from.on("close",       function()  {
    process.exitCode = 0;
    if (process.disconnect) process.disconnect();
    reallyExit(0);
  });

  let is_exiting = false;
  function close_worker_process() {
    if (is_exiting) return reallyExit(0);
    is_exiting = true;
    compute_core.emit_to("close");
    setTimeout(function() { reallyExit(0); }, 3000).unref();
  }
  process.on("SIGINT", close_worker_process);
  process.on("SIGTERM", close_worker_process);
  if (process.platform === "win32") process.on("SIGBREAK", close_worker_process);
  else process.on("SIGHUP", close_worker_process);

  function handle_msg(msg) {
    switch (msg.type) {
      case "job": case "bench": case "test":
        // find dev for this specific thread from msg.job.dev list
        msg.job.dev = module.exports.get_thread_dev(thread_id, msg.job.dev);
        msg.job.thread_id = thread_id;
        compute_core.emit_to(msg.type, msg.job);
        break;
      case "pause": case "close":
        compute_core.emit_to(msg.type);
        break;
      default: module.exports.log_err("Unknown thread message");
    }
  }

  // process messages from the master thread
  process.on("message", handle_msg);
  if (!process.send) {
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

  return true;
};

// get thread dev stripping ^thread specification from it
function parseThreadDev(dev_part) {
  const m = dev_part.match(/^([^^]+)(?:\^(\d+))?$/);
  return { dev: m ? m[1] : dev_part, threads: m && m[2] ? parseInt(m[2]) : 1 };
}

module.exports.is_valid_dev = function(dev) {
  return typeof dev === "string" && dev.split(",").every(function(dev_part) {
    return /^(?:cpu\d*|gpu\d+)(?:\*[1-9]\d*)?(?:\^[1-9]\d*)?$/.test(dev_part);
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
  return m ? parseInt(m[1]) : 1;
};

module.exports.messageWorkers = function(msg) {
  const targets = [];
  for (const worker_id of worker_ids) {
    const worker = worker_procs[worker_id];
    if (worker && worker.stdin && worker.stdin.writable) {
      if (msg.type === "close") worker.expectedClose = true;
      worker.stdin.write(JSON.stringify(msg) + "\n");
      targets.push({ type: "subprocess", id: worker_id, worker });
      continue;
    }

    const cluster_worker = cluster.workers[worker_id];
    if (cluster_worker) {
      if (msg.type === "close") cluster_worker.expectedClose = true;
      if (cluster_worker.isConnected && !cluster_worker.isConnected()) continue;
      try {
        cluster_worker.send(msg, function(error) {
          if (!error || msg.type === "close" || cluster_worker.expectedClose) return;
          cluster_worker.emit("error", error);
        });
      } catch (error) {
        if (msg.type !== "close" && !cluster_worker.expectedClose) cluster_worker.emit("error", error);
        continue;
      }
      targets.push({ type: "cluster", id: worker_id, worker: cluster_worker });
    }
  }
  return targets;
};

function forceCloseWorker(target) {
  const worker = target.worker;
  if (!worker) return;
  if (target.type === "subprocess") {
    if (worker.exitCode !== null || worker.signalCode !== null || worker.killed) return;
    if (is_windows_process && worker.pid) {
      const killer = childProcess.spawn("taskkill", ["/pid", String(worker.pid), "/t", "/f"], {
        stdio: "ignore",
      });
      killer.on("error", function() { worker.kill(); });
      return;
    }
    worker.kill("SIGKILL");
    return;
  }

  if (worker.isDead && worker.isDead()) return;
  worker.kill("SIGKILL");
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
    if (use_subprocess_workers) {
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
        messageHandler({
          type: "error",
          value: { message: "Worker " + i + " failed to start: " + error.message },
          thread_id: i
        });
      });
      thread.on("exit", function(code, signal) {
        if (worker_procs[i] !== thread) return;
        delete worker_procs[i];
        worker_ids = worker_ids.filter((worker_id) => worker_id !== i);
        if (thread.expectedClose) return;
        const detail = [];
        if (recentStdout.trim()) detail.push("stdout: " + recentStdout.trim());
        if (recentStderr.trim()) detail.push("stderr: " + recentStderr.trim());
        messageHandler({
          type: "error",
          value: {
            message: "Worker " + i + " exited unexpectedly" +
              (signal ? " with signal " + signal : " with code " + code) +
              (detail.length ? ". " + detail.join(" | ") : "")
          },
          thread_id: i
        });
      });
      worker_ids.push(i);
      worker_procs[i] = thread;
    } else {
      const thread = cluster.fork(env);
      thread.on("message", messageHandler);
      thread.on("error", function(error) {
        if (thread.expectedClose) return;
        messageHandler({
          type: "error",
          value: { message: "Worker " + i + " IPC error: " + error.message },
          thread_id: i
        });
      });
      thread.on("exit", function(code, signal) {
        worker_ids = worker_ids.filter((worker_id) => worker_id !== thread.id);
        if (thread.expectedClose) return;
        messageHandler({
          type: "error",
          value: {
            message: "Worker " + i + " exited unexpectedly" +
              (signal ? " with signal " + signal : " with code " + code)
          },
          thread_id: i
        });
      });
      worker_ids.push(thread.id);
    }
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

module.exports.diff2target = function(diff) {
  if (BigInt(diff) === 0n) return "0000000000000000";
  // Reverse of: div = BigInt(0xFFFFFFFFFFFFFFFF) / diff
  let div = BigInt("0xFFFFFFFFFFFFFFFF") / BigInt(diff);
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
  for (let i = 0; i < hex.length; i += 8) pow.push(parseInt(hex.slice(i, i + 8), 16));
  return pow;
}
