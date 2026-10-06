// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const h = require("./helper.js");
const compilerPolicy = require("./compiler-policy.js");
const gpuTuning = require("./gpu-tuning.js");
const {normalizeAlgoName} = require("./miner/algorithms");

const version_str = require("./package.json").version;
const momMultiAlgoProxyHost = "mom.moneroocean.stream";

/** @param {unknown} url @returns {boolean} */
function isMomMultiAlgoProxyUrl(url) {
  return typeof url === "string" &&
    url.toLowerCase().replace(/\.$/, "") === momMultiAlgoProxyHost;
}

/**
 * The option schema is intentionally dynamic: public keys describe nested options, while the
 * underscored keys carry template metadata consumed by the parser.
 * @typedef {UnknownRecord & {
 *   _help?: string,
 *   _template?: UnknownRecord,
 *   _array?: UnknownRecord[],
 *   _map?: UnknownRecord,
 * }} OptionSchema
 */

module.exports.agent_str = "mom v" + version_str;
const releaseCommandNames = new Set(["mom", "mom.exe"]);

/**
 * @param {string} url
 * @param {number} port
 * @param {boolean} is_tls
 * @param {string} login
 * @param {string} pass
 * @returns {PoolConfig}
 */
module.exports.pool_create = function(url, port, is_tls, login, pass) {
  return {
    url, port, is_tls, login, pass,
    protocol:          null,
    tls_verify:        false,
    is_nicehash:       url.includes("nicehash"),
    is_keepalive:      true,
    // JSON login permits the multi-algo MoM proxy; subscribe needs a preset algorithm.
    use_subscribe:     !isMomMultiAlgoProxyUrl(url),
    worker:            "",
    pearlhash_target_format: "base",
    socket:            null,
    keepalive:         null,
    pending_subscribe: false,
    pending_authorize: false,
    pending_submit_count: 0,
    logged_in:         false,
    last_connect_time: 0,
    last_job:          null,
    good_shares:       0,
    bad_shares:        0,
  };
};

const dev_help = 'device config "[<dev>[*MAIN|*[name=value;...]][^P],]+", dev = ' +
                 "{cpu, gpu<N>, cpu<N>}, N = device number, MAIN = algorithm primary tuning value, " +
                 "P = number of parallel processes";

module.exports.opt_help = {
  job: {
    _help:        "JSON string of the default job params (mostly used in test/bench mode)",
    algo:         [ null,  'algo name of the job (only used with "mine" directive)' ],
    dev:          [ "cpu", dev_help ],
    blob_hex: [ "0305A0DBD6BF05CF16E503F3A66F78007CBF34144332ECBFC22ED95C8700383B309ACE1923A0964B"
              + "00000008BA939A62724C0D7581FCE5761E9D8A0E6A1C3F924FDD8493D1115649C05EB601",
    "hexadecimal string of input blob" ],
    seed_hex: [ "3132333435363738393031323334353637383930313233343536373839303132",
      "hexadecimal string of seed hash blob (used for rx algos)" ],
    height:   [ 0, "Block height used by some algos"],
    backend:  [ "auto", "GPU implementation (see README backend table)" ],
    pearlhash_cert_version: [ undefined, "PearlHash certificate version (3)" ],
  },
  pool_time: {
    _help:             "JSON string of pool related timings (in seconds)",
    stats:             [ 10*60,  "time to show pool mining stats" ],
    connect_throttle:  [ 60,     "time between pool connection attempts" ],
    primary_reconnect: [ 90,     "time to try to use primary pool if currently on backup pool" ],
    first_job_wait:    [ 15,     "consider pool bad if no first job after connection" ],
    close_wait:        [ 10,     "keep pool socket to submit delayed jobs" ],
    donate_interval:   [ 100*60, "time before donation pool is activated" ],
    donate_length:     [ 1*60,   "donation pool work time" ],
    keepalive:         [ 5*60,   "interval to send keepalive messages" ],
  },
  pool: {
    _help: "add backup pool, defined by the following keys:",
    _template: {
      url:                [ undefined, "pool DNS or IP address" ],
      port:               [ undefined, "pool port" ],
      is_tls:             [ false, "is pool port is encrypted using TLS/SSL" ],
      protocol:           [ null, "pool protocol override (normally inferred from the algorithm)" ],
      tls_verify:         [ false, "verify pool TLS/SSL certificate" ],
      is_nicehash:        [ false, "nicehash nonce mining mode support" ],
      is_keepalive:       [ true, "sends keepalive messages to the pool to avoid disconnect" ],
      use_subscribe:      [ true, "use the algorithm's normal pool handshake; false forces XMR login with pushed jobs" ],
      worker:             [ "", "PearlHash/XELIS worker name (mining.authorize)" ],
      pearlhash_target_format: [ "base", "PearlHash subscribe target: base (scaled) or jackpot (final)" ],
      login:              [ undefined, "pool login data" ],
      pass:               [ "", "pool password" ],
      _socket:            [ null, "network socket object" ],
      _keepalive:         [ null, "keepalive timer object" ],
      _pending_subscribe: [ false, "subscribe request is waiting for response" ],
      _pending_authorize: [ false, "authorize request is waiting for response" ],
      _pending_submit_count: [ 0, "share submissions waiting for responses" ],
      _logged_in:         [ false, "pool login completed" ],
      _last_connect_time: [ 0, "last connect time for throttling purposes" ],
      _last_job:          [ null, "last job object" ],
      _good_shares:       [ 0, "number of accepted shares" ],
      _bad_shares:        [ 0, "number of invalid shares" ],
    },
    _array: [
      // The MoM donation proxy speaks the XMR `login` dialect.
      {
        ...module.exports.pool_create("mom.moneroocean.stream", 20001, true, "user", "pass"),
        use_subscribe: false,
      }
    ]
  },
  default_msr: {
    _help: "stores default MSR register values to restore them without reboot, " +
           "keys should be hex strings with 0x prefix",
    _template: {
      value:  [ undefined, "MSR register value in hex string with 0x prefix format" ],
      mask:   [ "0xFFFFFFFFFFFFFFFF", "MSR register mask in hex string with 0x prefix format" ],
    },
    _map: {}
  },
  pool_ids: {
    primary: null,
    donate:  0,
  },
  algo_param: {
    _help: "new algo params, defined by the following keys:",
    _template: {
      dev:      [ "cpu", dev_help ],
      perf:     [ null, "local algo hashrate (pool protocol normalization is automatic)" ],
      backend:  [ "auto", "GPU implementation (see README backend table)" ],
      tuning:   [ {}, "partial algorithm-specific GPU tuning object; omitted fields use auto heuristics" ],
    },
    _map: {}
  },
  log_level: [ 0, "log level: 0=minimal, 1=verbose, 2=network debug, 3=compute core debug" ],
  bench_algo_params: [ 1, "benchmark algo params before mining: 0=skip, 1=active MoneroOcean coin algos plus rx/2 and pearlhash, 2=all supported algos" ],
  gpu_tune: [ 0, "one-shot empirical tuning before missing GPU benchmarks: 0=off, 1=bounded search (slow)" ],
  save_config: [ "", "file name to save config in JSON format (only for mine directive)" ]
};

// Plain object from this process or a VM realm (excludes arrays, class instances, and null).
/** @param {unknown} value @returns {value is UnknownRecord} */
function isObject(value) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {return false;}
  const prototype = Object.getPrototypeOf(value);
  return prototype === null || Object.getPrototypeOf(prototype) === null;
}

/** @param {unknown} value @returns {value is OptionSchema} */
function isOptionSchema(value) {
  return isObject(value) &&
    (value["_help"] === undefined || typeof value["_help"] === "string") &&
    (value["_template"] === undefined || isObject(value["_template"])) &&
    (value["_array"] === undefined ||
      (Array.isArray(value["_array"]) && value["_array"].every(isObject))) &&
    (value["_map"] === undefined || isObject(value["_map"]));
}

/** @param {unknown} value @param {string} key @returns {OptionSchema | null} */
function nestedOptionSchema(value, key) {
  if (!isObject(value)) {return null;}
  if (!isOptionSchema(value)) {throw new Error(`Invalid option schema: ${key}`);}
  return value;
}

/** @param {unknown} value @returns {value is UnknownRecord[]} */
function isObjectArray(value) {
  return Array.isArray(value) && value.every(isObject);
}
const reservedKeys = new Set(["__proto__", "constructor", "prototype"]);
const jobJsonFields = new Set([
  "algo", "dev", "blob_hex", "seed_hex", "height", "backend", "target", "nonce",
  "nicehash_mask", "noncebytes", "nonceoffset", "proofsize", "pearlhash_cert_version",
]);
const jobIntegerFields = new Set([
  "height", "noncebytes", "nonceoffset", "proofsize", "pearlhash_cert_version",
]);
const poolBooleanFields = ["is_tls", "tls_verify", "is_nicehash", "is_keepalive", "use_subscribe"];
const poolRuntimeFields = new Set(
  Object.keys(module.exports.opt_help.pool._template)
    .filter((key) => key.startsWith("_")).map((key) => key.slice(1))
);
const poolProtocols = new Set([
  "login", "raven", "eth", "erg", "pearlhash", "zelhash", "verthash",
  "kaspa", "hoosat", "beam", "ironfish", "xelis", "conflux", "echelon", "cortex",
]);
const boundedIntegerOptions = new Map([
  ["log_level", 3],
  ["bench_algo_params", 2],
  ["gpu_tune", 1],
]);
const maxTimerSeconds = 0x7fffffff / 1000;
const timerOptions = new Set([
  "pool_time.stats", "pool_time.connect_throttle", "pool_time.primary_reconnect",
  "pool_time.first_job_wait", "pool_time.close_wait", "pool_time.donate_interval",
  "pool_time.donate_length", "pool_time.keepalive",
]);
const positiveTimerOptions = new Set([
  "pool_time.stats", "pool_time.primary_reconnect", "pool_time.first_job_wait",
  "pool_time.donate_interval", "pool_time.donate_length", "pool_time.keepalive",
]);
/** @type {Record<string, (arg: string, parsed: UnknownRecord, value: UnknownRecord) => UnknownRecord>} */
const templateValidators = {
  pool: validatePoolOption,
  default_msr: validateDefaultMsrOption,
  algo_param: validateAlgoParamOption,
};

/** @param {unknown} val @returns {unknown} */
function cloneDefault(val) {
  if (Array.isArray(val)) {
    return val.map(cloneDefault);
  }
  if (!isObject(val)) {
    return val;
  }
  /** @type {UnknownRecord} */
  const cloned = {};
  for (const [key, value] of Object.entries(val)) {
    cloned[key] = cloneDefault(value);
  }
  return cloned;
}

/** @param {string} key */
function isPublicKey(key) {
  return !key.startsWith("_");
}

/** @param {object} object */
function publicKeys(object) {
  return Object.keys(object).filter(isPublicKey);
}

/** @param {unknown} value */
function numberValue(value) {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string" && /^-?\d+(?:\.\d+)?$/.test(value)) {
    return Number(value);
  }
  return Number.NaN;
}

/** @param {UnknownRecord} pool */
function validatePool(pool) {
  if (typeof pool["url"] !== "string" || pool["url"] === "") {
    return "url must be a non-empty string";
  }
  const port = numberValue(pool["port"]);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    return "port must be an integer from 1 to 65535";
  }
  pool["port"] = port;
  const invalidBoolean = poolBooleanFields.find((key) => typeof pool[key] !== "boolean");
  if (invalidBoolean) {
    return invalidBoolean + " must be boolean";
  }
  for (const key of ["login", "pass", "worker"]) {
    if (typeof pool[key] !== "string") {
      return key + " must be a string";
    }
  }
  if (pool["pearlhash_target_format"] !== "base" && pool["pearlhash_target_format"] !== "jackpot") {
    return "pearlhash_target_format must be base or jackpot";
  }
  if (pool["protocol"] === null) {
    return null;
  }
  if (typeof pool["protocol"] !== "string") {
    return "protocol must be a string or null";
  }
  const protocol = pool["protocol"].toLowerCase();
  pool["protocol"] = protocol;
  return poolProtocols.has(protocol) ? null : "protocol is not supported: " + protocol;
}

/** @param {unknown} def_val */
function formatDefaultValue(def_val) {
  switch (typeof def_val) {
    case "string":
      return "\"" + def_val + "\"";
    case "bigint":
      return "0x" + def_val.toString(16);
    case "object":
      return JSON.stringify(def_val);
    default:
      return def_val;
  }
}

/** @param {unknown} def_val */
function defaultSuffix(def_val) {
  return typeof def_val !== "undefined" ? " (" + formatDefaultValue(def_val) + " by default)" : "";
}

/** @param {string} arg @param {string} val @returns {UnknownRecord} */
function parseJsonObject(arg, val) {
  let parsed;
  try {
    parsed = JSON.parse(val);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return module.exports.print_help("Can't parse option " + arg + " JSON param: " + message);
  }
  if (!isObject(parsed)) {
    return module.exports.print_help("Option " + arg + " JSON param must be an object");
  }
  const algo = parsed["algo"];
  if (algo !== undefined && algo !== null && typeof algo !== "string") {
    return module.exports.print_help("Option " + arg + " has invalid algo value");
  }
  for (const field of ["blob_hex", "seed_hex", "target", "nicehash_mask"]) {
    if (Object.hasOwn(parsed, field) && typeof parsed[field] !== "string") {
      return module.exports.print_help("Option " + arg + " has invalid " + field + " value");
    }
  }
  if (Object.hasOwn(parsed, "dev") && !h.is_valid_dev(parsed["dev"], algo || "")) {
    return module.exports.print_help("Option " + arg + " has invalid dev value: " + parsed["dev"]);
  }
  return parsed;
}

/** @param {string} arg @param {unknown} val */
function parseNonNegativeNumber(arg, val) {
  const num = numberValue(val);
  if (!Number.isFinite(num)) {
    return module.exports.print_help("Option " + arg + " param must be a number: " + val);
  }
  if (num < 0) {
    return module.exports.print_help("Option " + arg + " param must be non-negative: " + val);
  }
  return num;
}

/** @param {string} arg @param {unknown} val @param {string} keyPath */
function parseOptionNumber(arg, val, keyPath) {
  const num = parseNonNegativeNumber(arg, val);
  if (positiveTimerOptions.has(keyPath) && num <= 0) {
    return module.exports.print_help("Option " + arg + " param must be positive: " + val);
  }
  if (timerOptions.has(keyPath) && num > maxTimerSeconds) {
    return module.exports.print_help(
      "Option " + arg + " param exceeds the maximum timer delay: " + val
    );
  }
  return num;
}

/** @param {string} arg @param {unknown} val */
function parseNonNegativeInteger(arg, val) {
  const num = parseNonNegativeNumber(arg, val);
  if (!Number.isSafeInteger(num)) {
    return module.exports.print_help("Option " + arg + " param must be a safe integer: " + val);
  }
  return num;
}

/** @param {string} arg @param {unknown} val */
function parsePearlHashCertVersion(arg, val) {
  const version = parseNonNegativeInteger(arg, val);
  if (version !== 3) {
    return module.exports.print_help(
      "Option " + arg + " only supports PearlHash certificate version 3: " + val
    );
  }
  return version;
}

/** @param {string} arg @param {unknown} val @param {number} max */
function parseBoundedInteger(arg, val, max) {
  const num = parseNonNegativeInteger(arg, val);
  if (num > max) {
    return module.exports.print_help("Option " + arg + " param must be at most " + max + ": " + val);
  }
  return num;
}

/** @param {unknown} help */
function isNumberOption(help) {
  return Array.isArray(help) && typeof help[0] === "number";
}

/** @param {string} arg @param {string} key_path_str @param {string} new_str_prefix */
function isJsonOptionArg(arg, key_path_str, new_str_prefix) {
  return arg === "--" + key_path_str ||
         arg === "--add." + key_path_str ||
         arg.startsWith(new_str_prefix);
}

/**
 * @param {string} arg
 * @param {string} key
 * @param {UnknownRecord} template
 * @param {UnknownRecord} values
 * @returns {UnknownRecord}
 */
function templateDefaults(arg, key, template, values) {
  const unknown = Object.keys(values).find((field) =>
    !isPublicKey(field) || !Object.hasOwn(template, field));
  if (unknown) {
    return module.exports.print_help(`Option ${arg} has unsupported field: ${unknown}`);
  }
  /** @type {UnknownRecord} */
  const result = {};
  for (const key2 of publicKeys(template)) {
    const definition = template[key2];
    if (!Array.isArray(definition)) {
      return module.exports.print_help(`Option ${arg} has invalid template field: ${key2}`);
    }
    const def_val = definition[0];
    // template keys without a default (undefined) are required, so they must be supplied
    if (typeof def_val === "undefined" && !Object.hasOwn(values, key2)) {
      return module.exports.print_help("Need to specify key value \"" + key2 + "\" in " + key + " JSON");
    }
    result[key2] = Object.hasOwn(values, key2) ? values[key2] : def_val;
  }
  return result;
}

// Validators receive both `parsed` (the raw JSON the user passed) and `value` (parsed merged with
// template defaults). Errors report from `parsed` so the message shows the user's original input,
// not a value that coercion may have already turned into NaN.
/** @param {string} arg @param {UnknownRecord} parsed @param {UnknownRecord} value */
function validatePoolOption(arg, parsed, value) {
  const err = validatePool(value);
  if (err) {
    return module.exports.print_help("Option " + arg + " has invalid pool " + err);
  }
  if (!Object.hasOwn(parsed, "use_subscribe")) {
    value["use_subscribe"] = !isMomMultiAlgoProxyUrl(value["url"]);
  }
  return value;
}

/** @param {string} arg @param {string} option */
function mapNameFromOption(arg, option) {
  const marker = "." + option + ".";
  const position = arg.indexOf(marker);
  const name = position === -1 ? "" : arg.slice(position + marker.length);
  if (option !== "algo_param") {return name;}
  const normalized = normalizeAlgoName(name);
  if (!normalized) {
    return module.exports.print_help("Option " + arg + " has invalid algorithm name");
  }
  return normalized;
}

/** @param {string} arg @param {UnknownRecord} parsed @param {UnknownRecord} value */
function validateDefaultMsrOption(arg, parsed, value) {
  const register = mapNameFromOption(arg, "default_msr");
  if (!/^0x[0-9a-f]{1,8}$/i.test(register)) {
    return module.exports.print_help("Option " + arg + " has invalid MSR register");
  }
  for (const field of ["value", "mask"]) {
    if (typeof value[field] !== "string" || !/^0x[0-9a-f]{1,16}$/i.test(value[field])) {
      return module.exports.print_help(`Option ${arg} has invalid MSR ${field}: ${parsed[field]}`);
    }
  }
  return value;
}

/** @param {string} arg @param {UnknownRecord} parsed @param {UnknownRecord} value */
function validateAlgoPerf(arg, parsed, value) {
  if (value["perf"] === null) {
    return value;
  }
  const perf = numberValue(value["perf"]);
  value["perf"] = perf;
  if (!Number.isFinite(perf) || perf < 0) {
    return module.exports.print_help("Option " + arg + " has invalid perf value: " + parsed["perf"]);
  }
  return value;
}

/** @param {string} arg @param {UnknownRecord} parsed @param {UnknownRecord} value */
function validateGpuBackend(arg, parsed, value) {
  if (typeof value["backend"] !== "string") {
    return module.exports.print_help(
      "Option " + arg + " has invalid backend value: " + parsed["backend"]
    );
  }
  const backend = value["backend"].toLowerCase();
  value["backend"] = backend;
  if (!compilerPolicy.validBackends.has(backend)) {
    return module.exports.print_help(
      "Option " + arg + " has invalid backend value: " + parsed["backend"]
    );
  }
  return value;
}

/** @param {string} arg @param {UnknownRecord} parsed @param {UnknownRecord} value */
function validateAlgoParamOption(arg, parsed, value) {
  const algo = mapNameFromOption(arg, "algo_param");
  try {
    gpuTuning.parseDeviceList(value["dev"], algo);
    const tuning = value["tuning"] === undefined ? {} : value["tuning"];
    value["tuning"] = gpuTuning.validateTuning(algo, tuning, `${arg}.tuning`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return module.exports.print_help("Option " + arg + " has invalid GPU tuning: " + message);
  }
  return validateGpuBackend(arg, parsed, validateAlgoPerf(arg, parsed, value));
}

/**
 * @param {string} arg
 * @param {string} key_path_str
 * @param {UnknownRecord} parsed
 * @param {UnknownRecord} value
 */
function validateTemplateValue(arg, key_path_str, parsed, value) {
  const validator = templateValidators[key_path_str];
  return validator ? validator(arg, parsed, value) : value;
}

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {OptionSchema} key_help
 * @param {string} key_path_str
 * @param {string} arg
 * @param {UnknownRecord} val
 */
function addArrayTemplateOption(opt, key, key_help, key_path_str, arg, val) {
  if (!("_array" in key_help) || arg !== "--add." + key_path_str) {
    return false;
  }
  const values = opt[key + "s"];
  if (!Array.isArray(values)) {return module.exports.print_help(`Option ${arg} target is not an array`);}
  values.push(val);
  return true;
}

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {OptionSchema} key_help
 * @param {string} new_str_prefix
 * @param {string} arg
 * @param {UnknownRecord} val
 */
function addMapTemplateOption(opt, key, key_help, new_str_prefix, arg, val) {
  if (!("_map" in key_help) || !arg.startsWith(new_str_prefix)) {
    return false;
  }
  const key2 = mapNameFromOption(arg, key);
  if (!key2 || reservedKeys.has(key2)) {
    return module.exports.print_help("Invalid option map key: " + key2);
  }
  const values = opt[key + "s"];
  if (!isObject(values)) {return module.exports.print_help(`Option ${arg} target is not a map`);}
  values[key2] = val;
  return true;
}

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {OptionSchema} key_help
 * @param {string} key_path_str
 * @param {string} new_str_prefix
 * @param {string} arg
 * @param {UnknownRecord} parsed
 */
function applyTemplateOption(opt, key, key_help, key_path_str, new_str_prefix, arg, parsed) {
  if (!key_help._template) {return module.exports.print_help(`Option ${arg} template is missing`);}
  const defaults = templateDefaults(arg, key, key_help._template, parsed);
  const value = validateTemplateValue(arg, key_path_str, parsed, defaults);
  return addArrayTemplateOption(opt, key, key_help, key_path_str, arg, value) ||
         addMapTemplateOption(opt, key, key_help, new_str_prefix, arg, value);
}

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {OptionSchema} key_help
 * @param {string} arg
 * @param {UnknownRecord} parsed
 */
function applyJsonOption(opt, key, key_help, arg, parsed) {
  const target = opt[key];
  if (!isObject(target)) {return module.exports.print_help(`Option ${arg} target is not an object`);}
  for (const key2 of Object.keys(parsed)) {
    const supported = key === "job" ? jobJsonFields.has(key2) :
      isPublicKey(key2) && Object.hasOwn(key_help, key2);
    if (!supported) {
      return module.exports.print_help(`Option ${arg} has unsupported field: ${key2}`);
    }
    const help = key_help[key2];
    let value = parsed[key2];
    if (key === "job" && key2 === "pearlhash_cert_version") {
      value = parsePearlHashCertVersion(arg + "." + key2, value);
    } else if (jobIntegerFields.has(key2)) {
      value = parseNonNegativeInteger(arg + "." + key2, value);
    } else if (isNumberOption(help)) {
      value = parseOptionNumber(arg + "." + key2, value, key + "." + key2);
    }
    target[key2] = value;
  }
  if (key === "job" && Object.hasOwn(parsed, "dev")) {
    target["dev_request"] = parsed["dev"];
  }
  if (key === "job" && Object.hasOwn(parsed, "backend")) {
    try {
      target["backend"] = compilerPolicy.validateBackend(target["backend"]);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return module.exports.print_help("Option " + arg + " " + message);
    }
  }
  return true;
}

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {OptionSchema} key_help
 * @param {string} key_path_str
 * @param {string} new_str_prefix
 * @param {string} arg
 * @param {string} val
 */
function parseJsonOption(opt, key, key_help, key_path_str, new_str_prefix, arg, val) {
  const parsed = parseJsonObject(arg, val);
  if (key_help._template) {
    return applyTemplateOption(opt, key, key_help, key_path_str, new_str_prefix, arg, parsed);
  }
  return applyJsonOption(opt, key, key_help, arg, parsed);
}

// set opt object based on default values from opt_help object
/** @param {object} opt @param {OptionSchema} opt_help */
module.exports.set_default_opts = function(opt, opt_help) {
  if (!isObject(opt)) {throw new Error("Default options target must be a plain object");}
  for (const key of publicKeys(opt_help)) {
    const key_help = opt_help[key];
    const schema = nestedOptionSchema(key_help, key);
    if (schema) {
      setObjectDefault(opt, key, schema);
    } else {
      opt[key] = simpleDefault(key_help);
    }
  }
};

/** @param {unknown} value @returns {value is PoolConfig} */
function isDefaultPool(value) {
  return isObject(value) && typeof value["url"] === "string" &&
    typeof value["port"] === "number" && typeof value["is_tls"] === "boolean" &&
    typeof value["login"] === "string" && typeof value["pass"] === "string" &&
    typeof value["is_keepalive"] === "boolean" && typeof value["is_nicehash"] === "boolean" &&
    typeof value["use_subscribe"] === "boolean" && typeof value["tls_verify"] === "boolean" &&
    (value["pearlhash_target_format"] === "base" || value["pearlhash_target_format"] === "jackpot") &&
    typeof value["worker"] === "string" && value["socket"] === null && value["keepalive"] === null &&
    value["last_job"] === null && typeof value["last_connect_time"] === "number" &&
    typeof value["pending_subscribe"] === "boolean" &&
    typeof value["pending_authorize"] === "boolean" &&
    typeof value["pending_submit_count"] === "number" &&
    Number.isSafeInteger(value["pending_submit_count"]) && value["pending_submit_count"] >= 0 &&
    typeof value["logged_in"] === "boolean" &&
    typeof value["good_shares"] === "number" && typeof value["bad_shares"] === "number";
}

/** @param {unknown} value @returns {value is MinerOptions} */
function isDefaultMinerOptions(value) {
  if (!isObject(value) || !isObject(value["job"]) || !isObject(value["pool_time"]) ||
      !isObject(value["pool_ids"]) || !isObject(value["algo_params"]) ||
      !isObject(value["default_msrs"]) || !Array.isArray(value["pools"]) ||
      !value["pools"].every(isDefaultPool)) {
    return false;
  }
  const job = value["job"];
  const times = value["pool_time"];
  const ids = value["pool_ids"];
  return (job["algo"] === null || typeof job["algo"] === "string") &&
    typeof job["dev"] === "string" && typeof job["backend"] === "string" &&
    typeof job["blob_hex"] === "string" && typeof job["seed_hex"] === "string" &&
    typeof job["height"] === "number" &&
    ["stats", "connect_throttle", "primary_reconnect", "first_job_wait", "close_wait",
      "donate_interval", "donate_length", "keepalive"].every((key) => typeof times[key] === "number") &&
    typeof ids["active"] === "number" &&
    (ids["primary"] === null || typeof ids["primary"] === "number") &&
    (ids["donate"] === null || typeof ids["donate"] === "number") &&
    typeof value["bench_algo_params"] === "number" && typeof value["gpu_tune"] === "number" &&
    typeof value["log_level"] === "number" && typeof value["save_config"] === "string";
}

/** @returns {MinerOptions} */
module.exports.create_default_opts = function() {
  /** @type {UnknownRecord} */
  const options = {};
  module.exports.set_default_opts(options, module.exports.opt_help);
  const poolIds = options["pool_ids"];
  if (!isObject(poolIds)) {throw new Error("Default pool IDs are malformed");}
  poolIds["active"] = 0;
  if (!isDefaultMinerOptions(options)) {throw new Error("Default options are malformed");}
  return options;
};

/** @param {unknown} key_help */
function simpleDefault(key_help) {
  return Array.isArray(key_help) ? cloneDefault(key_help[0]) : key_help;
}

/** @param {UnknownRecord} opt @param {string} key @param {OptionSchema} key_help */
function setObjectDefault(opt, key, key_help) {
  // _array/_map templates seed the pluralized collection; plain objects recurse
  if ("_array" in key_help) {
    opt[key + "s"] = cloneDefault(key_help._array);
  } else if ("_map" in key_help) {
    opt[key + "s"] = cloneDefault(key_help._map);
  } else {
    module.exports.set_default_opts(opt[key] = {}, key_help);
  }
}

/** @param {MinerOptions} opt */
module.exports.saved_config = function(opt) {
  const {
    job: _job,
    save_config: _saveConfig,
    ...saved
  } = opt;
  const {active: _active, ...pool_ids} = opt.pool_ids;
  return {
    ...saved,
    pools: opt.pools.map(durablePoolFields),
    pool_ids,
  };
};

/** @param {MinerOptions} opt */
module.exports.redacted_options = function(opt) {
  return {
    ...opt,
    pools: opt.pools.map((pool) => ({...pool, login: "<redacted>", pass: "<redacted>"})),
  };
};

/** @param {string} message @returns {never} */
function configError(message) {
  return module.exports.print_help("Invalid config: " + message);
}

/** @param {MinerOptions} opt @param {string} arg @param {unknown} value */
function applyConfigOption(opt, arg, value) {
  const encoded = typeof value === "string" ? value : JSON.stringify(value);
  if (encoded === undefined) {return configError("unsupported undefined value: " + arg);}
  if (!module.exports.parse_opt(opt, module.exports.opt_help, arg, encoded, "")) {
    return configError("unsupported field: " + arg.replace(/^--/, ""));
  }
}

/** @param {MinerOptions} opt @param {"default_msrs" | "algo_params"} key
 * @param {string} option @param {unknown} values */
function applyConfigMap(opt, key, option, values) {
  if (!isObject(values)) {
    return configError(key + " must be an object");
  }
  opt[key] = {};
  for (const [name, value] of Object.entries(values)) {
    if (!name || reservedKeys.has(name)) {
      return configError("invalid " + key + " key: " + name);
    }
    applyConfigOption(opt, "--new." + option + "." + name, value);
  }
}

/** @param {unknown} value @returns {UnknownRecord} */
function durablePoolFields(value) {
  if (!isObject(value)) {
    return configError("pool must be an object");
  }
  return Object.fromEntries(Object.entries(value).filter(([key]) =>
    isPublicKey(key) && Object.hasOwn(module.exports.opt_help.pool._template, key)));
}

/** Retain unknown fields so the option parser reports typos; discard only legacy runtime state.
 * @param {unknown} value @returns {UnknownRecord} */
function configPoolFields(value) {
  if (!isObject(value)) {
    return configError("pool must be an object");
  }
  return Object.fromEntries(Object.entries(value).filter(([key]) => !poolRuntimeFields.has(key)));
}

/** @param {MinerOptions} opt @param {unknown} value */
function applyPoolIds(opt, value) {
  if (!isObject(value)) {
    return configError("pool_ids must be an object");
  }
  const unknown = Object.keys(value).find((key) => key !== "primary" && key !== "donate");
  if (unknown) {
    return configError("unsupported field: pool_ids." + unknown);
  }
  for (const [key, id] of Object.entries(value)) {
    if (key === "donate" && id === null) {
      opt.pool_ids.donate = null;
      continue;
    }
    if (typeof id !== "number" || !Number.isSafeInteger(id) || id < 0) {
      return configError("pool_ids." + key + " must be a non-negative integer");
    }
    if (key === "primary") {opt.pool_ids.primary = id;}
    if (key === "donate") {opt.pool_ids.donate = id;}
  }
}

/** @param {MinerOptions} opt */
function validateConfigPoolIds(opt) {
  const validId = (/** @type {unknown} */ id) =>
    typeof id === "number" && Number.isSafeInteger(id) && id >= 0 && id < opt.pools.length;
  if (!validId(opt.pool_ids.primary)) {
    return configError("pool_ids.primary does not select a pool");
  }
  if (opt.pool_ids.donate !== null && !validId(opt.pool_ids.donate)) {
    return configError("pool_ids.donate does not select a pool");
  }
  if (opt.pool_ids.primary === opt.pool_ids.donate) {
    return configError("pool_ids.primary and pool_ids.donate must select different pools");
  }
}

/** @param {MinerOptions} opt @param {unknown} values */
module.exports.apply_config = function(opt, values) {
  if (!isObject(values)) {
    return configError("root must be an object");
  }
  for (const [key, value] of Object.entries(values)) {
    switch (key) {
      case "job":
      case "pool_time":
        applyConfigOption(opt, "--" + key, value);
        break;
      case "pools":
        if (!Array.isArray(value)) {
          return configError("pools must be an array");
        }
        opt.pools = [];
        for (const pool of value) {
          applyConfigOption(opt, "--add.pool", configPoolFields(pool));
        }
        break;
      case "default_msrs":
        applyConfigMap(opt, key, "default_msr", value);
        break;
      case "algo_params":
        applyConfigMap(opt, key, "algo_param", value);
        break;
      case "pool_ids":
        applyPoolIds(opt, value);
        break;
      case "log_level":
      case "bench_algo_params":
      case "gpu_tune":
      case "save_config":
        if (isObject(value) || Array.isArray(value) || value === null) {
          return configError(key + " must be a scalar");
        }
        applyConfigOption(opt, "--" + key, value);
        break;
      default: return configError("unsupported field: " + key);
    }
  }
  validateConfigPoolIds(opt);
};

/** @param {MinerOptions} opt @param {string} configFile */
module.exports.load_config = function(opt, configFile) {
  const filename = path.resolve(configFile);
  h.log("Loading config file " + filename);
  let values;
  try {
    values = JSON.parse(fs.readFileSync(filename, "utf8"));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return module.exports.print_help("Can't load config file " + filename + ": " + message);
  }
  module.exports.apply_config(opt, values);
};

/** @param {string} file */
module.exports.is_config_file = function(file) {
  return path.extname(file).toLowerCase() === ".json";
};

// prints options help from opt_help
/** @param {OptionSchema} opt_help @param {string} depth_str @param {string} base_key_path_str */
function print_opt_help(opt_help, depth_str, base_key_path_str) {
  for (const key of publicKeys(opt_help)) {
    const key_path_str = base_key_path_str ? base_key_path_str + "." + key : key;
    const definition = opt_help[key];
    const schema = nestedOptionSchema(definition, key_path_str);
    if (schema) {
      printObjectOptHelp(schema, depth_str, key_path_str);
    } else {
      printSimpleOptHelp(definition, depth_str, key_path_str);
    }
  }
}

/** @param {string} line @param {string} help */
function paddedHelp(line, help) {
  return line.padEnd(36, " ") + help;
}

/** @param {OptionSchema} key_help @param {string} depth_str @param {string} key_path_str */
function printTemplateHeader(key_help, depth_str, key_path_str) {
  const fields = " '{[\"<key>\": <value>,]+}': ";
  if ("_array" in key_help) {
    console.log(paddedHelp(depth_str + "--add." + key_path_str + fields, key_help._help || ""));
  } else if ("_map" in key_help) {
    console.log(paddedHelp(
      depth_str + "--new." + key_path_str + ".<name>" + fields, key_help._help || ""));
  }
}

/** @param {UnknownRecord} template @param {string} depth_str */
function printTemplateFields(template, depth_str) {
  for (const key of publicKeys(template)) {
    const definition = template[key];
    if (!Array.isArray(definition) || typeof definition[1] !== "string") {
      throw new Error(`Invalid option help template field: ${key}`);
    }
    const def_val = definition[0];
    if (def_val === null) {
      continue;
    } // do not show internal params
    console.log(paddedHelp(depth_str + "  " + key + ": ",
      definition[1] + defaultSuffix(def_val)));
  }
}

/** @param {OptionSchema} key_help @param {string} depth_str @param {string} key_path_str */
function printObjectOptHelp(key_help, depth_str, key_path_str) {
  if (typeof key_help._help === "undefined") {
    return;
  }
  if ("_template" in key_help) {
    printTemplateHeader(key_help, depth_str, key_path_str);
    printTemplateFields(key_help._template, depth_str);
  } else {
    console.log(paddedHelp(depth_str + "--" + key_path_str + " '{...}': ", key_help._help));
    print_opt_help(key_help, depth_str + "  ", key_path_str);
  }
  console.log();
}

/** @param {unknown} key_help @param {string} depth_str @param {string} key_path_str */
function printSimpleOptHelp(key_help, depth_str, key_path_str) {
  if (!Array.isArray(key_help) || typeof key_help[1] !== "string") {
    throw new Error(`Invalid option help field: ${key_path_str}`);
  }
  const def_val = key_help[0];
  if (def_val === null) {
    return;
  } // do not show internal params
  console.log(paddedHelp(depth_str + "--" + key_path_str + ": ", key_help[1] + defaultSuffix(def_val)));
}

function helpCommand() {
  if (process.env["MOM_COMMAND"]) {
    return process.env["MOM_COMMAND"];
  }
  const exe = path.basename(process.argv[1] || "");
  return releaseCommandNames.has(exe) ? "./" + exe : "node mom.js";
}

/** @param {string | undefined} err_str @returns {never} */
function finishHelp(err_str) {
  if (err_str) {
    h.log_err(err_str);
  }
  process.exit(err_str ? 1 : 0);
}

/** @param {string} [err_str] @returns {never} */
module.exports.print_help = function(err_str) {
  const str = `
# Node.js/SYCL based CPU/GPU miner v${version_str}
$ ${helpCommand()} <directive> <parameter>+ [<option>+]

Directives:
  mine  (<pool_address:port[tls]> <login> [<pass>]|<config.json>)
  test  <algo> <result_hash_hex_str>
  bench <algo>
  algorithms  List supported algorithms and automatic device settings

Options:`;
  console.log(str);
  print_opt_help(this.opt_help, "", "");
  finishHelp(err_str);
};

// recursively parses all options specified by the opt_help data structure
/**
 * @param {object} opt
 * @param {OptionSchema} opt_help
 * @param {string} arg
 * @param {string} val
 * @param {string} base_key_path_str
 */
module.exports.parse_opt = function(opt, opt_help, arg, val, base_key_path_str) {
  if (!isObject(opt)) {throw new Error("Option target must be a plain object");}
  for (const key of publicKeys(opt_help)) {
    const key_help = opt_help[key];
    const key_path_str = base_key_path_str ? base_key_path_str + "." + key : key;
    if (parseOptionEntry(opt, key, key_help, key_path_str, arg, val)) {
      return true;
    }
  }
  return false;
};

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {OptionSchema} key_help
 * @param {string} key_path_str
 * @param {string} arg
 * @param {string} val
 */
function parseObjectOption(opt, key, key_help, key_path_str, arg, val) {
  if (!("_help" in key_help)) {
    return false;
  }
  const new_str_prefix = "--new." + key_path_str + ".";
  // consider val as JSON string here
  if (isJsonOptionArg(arg, key_path_str, new_str_prefix)) {
    return parseJsonOption(opt, key, key_help, key_path_str, new_str_prefix, arg, val);
  }
  const target = opt[key];
  if (!isObject(target)) {return false;}
  return module.exports.parse_opt(target, key_help, arg, val, key_path_str);
}

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {unknown} key_help
 * @param {string} key_path_str
 * @param {string} arg
 * @param {string} val
 */
function parseSimpleOption(opt, key, key_help, key_path_str, arg, val) {
  if (arg !== "--" + key_path_str) {
    return false;
  }
  if (key_path_str === "job.pearlhash_cert_version") {
    opt[key] = parsePearlHashCertVersion(arg, val);
  } else {
    const max = boundedIntegerOptions.get(key_path_str);
    opt[key] = max === undefined ?
      (isNumberOption(key_help) ? parseOptionNumber(arg, val, key_path_str) : val) :
      parseBoundedInteger(arg, val, max);
  }
  if (key_path_str === "job.dev") {opt["dev_request"] = val;}
  return true;
}

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {unknown} key_help
 * @param {string} key_path_str
 * @param {string} arg
 * @param {string} val
 */
function parseOptionEntry(opt, key, key_help, key_path_str, arg, val) {
  const schema = nestedOptionSchema(key_help, key_path_str);
  if (schema) {
    return parseObjectOption(opt, key, schema, key_path_str, arg, val);
  }
  return parseSimpleOption(opt, key, key_help, key_path_str, arg, val);
}

// inject internal default values to opt object from opt_help object
/** @param {object} opt @param {OptionSchema} opt_help */
module.exports.set_internal_opts = function(opt, opt_help) {
  if (!isObject(opt)) {throw new Error("Internal options target must be a plain object");}
  for (const key of publicKeys(opt_help)) {
    const definition = opt_help[key];
    const schema = nestedOptionSchema(definition, key);
    if (schema) {setInternalObject(opt, key, schema);}
  }
};

/** @param {UnknownRecord} opt @param {string} key @param {OptionSchema} key_help */
function templateItems(opt, key, key_help) {
  const collection = opt[key + "s"];
  let values;
  if ("_array" in key_help) {
    if (!Array.isArray(collection)) {
      throw new Error(`Invalid internal option collection: ${key}`);
    }
    values = collection;
  } else if ("_map" in key_help) {
    if (!isObject(collection)) {
      throw new Error(`Invalid internal option collection: ${key}`);
    }
    values = Object.values(collection);
  } else {
    throw new Error(`Internal option template has no collection: ${key}`);
  }
  if (!isObjectArray(values)) {
    throw new Error(`Invalid internal option collection: ${key}`);
  }
  return values;
}

/** @param {UnknownRecord} item @param {UnknownRecord} template */
function applyInternalTemplateValues(item, template) {
  for (const [key, val] of Object.entries(template)) {
    if (key.startsWith("_")) {
      if (!Array.isArray(val)) {throw new Error(`Invalid internal option field: ${key}`);}
      item[key.substring(1)] = val[0];
    }
  }
}

/** @param {UnknownRecord} opt @param {string} key @param {OptionSchema} key_help */
function setInternalObject(opt, key, key_help) {
  if (!key_help._template) {
    const target = opt[key];
    if (!isObject(target)) {throw new Error(`Invalid internal option object: ${key}`);}
    return module.exports.set_internal_opts(target, key_help);
  }
  for (const item of templateItems(opt, key, key_help)) {
    applyInternalTemplateValues(item, key_help._template);
  }
}
