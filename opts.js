// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const h = require("./helper.js");
const compilerPolicy = require("./compiler-policy.js");
const gpuTuning = require("./gpu-tuning.js");
const {normalizeAlgoName, POOL_PROTOCOL_BY_ALGO} = require("./miner/algorithms");
const {normalizeFingerprint: normalizeTlsFingerprint} = require("./pool/tls");
const {errorMessage, isReservedKey, isSafeIntegerAtLeast} = require("./helper/validate");

const VERSION = require("./package.json").version;
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

module.exports.agent_str = "mom v" + VERSION;
const releaseCommandNames = new Set(["mom", "mom.exe"]);

// Pool fields with their defaults and help. Public keys are pool options (undefined marks a
// required one); "_" keys are per-connection runtime state that set_internal_opts resets.
const poolTemplate = {
  url:                [ undefined, "pool DNS or IP address" ],
  port:               [ undefined, "pool port" ],
  is_tls:             [ false, "is pool port is encrypted using TLS/SSL" ],
  protocol:           [ null, "pool protocol override (normally inferred from the algorithm)" ],
  tls_verify:         [ false, "verify pool TLS/SSL certificate" ],
  tls_fingerprint:    [ "", "pin TLS leaf certificate SHA-256 (64 hex digits, optional colons)" ],
  is_nicehash:        [ false, "nicehash nonce mining mode support; nicehash hosts default to true" ],
  is_keepalive:       [ true, "sends keepalive messages to the pool to avoid disconnect" ],
  use_subscribe:      [ true, "use the algorithm's normal pool handshake; false forces XMR login with pushed jobs" +
    " (the MoM proxy host defaults to false)" ],
  worker:             [ "", "PearlHash/XELIS worker name (mining.authorize); overrides a login's .worker suffix" ],
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
  _job_state:         [ {}, "negotiated job controls" ],
  _good_shares:       [ 0, "number of accepted shares" ],
  _bad_shares:        [ 0, "number of invalid shares" ],
};

// Defaults that follow from the pool host; a field set explicitly in a pool option wins. JSON login
// permits the multi-algo MoM proxy (subscribe needs a preset algorithm), and NiceHash assigns the
// nonce's high byte, so its hosts need nicehash mode. Both the positional pool and --add.pool
// (including config-file pools) use these.
/** @param {unknown} url */
function hostPoolDefaults(url) {
  return {
    is_nicehash: typeof url === "string" && url.toLowerCase().includes("nicehash"),
    use_subscribe: !isMomMultiAlgoProxyUrl(url),
  };
}

// A pool with every field at its template default, fresh runtime state, and the host defaults.
/**
 * @param {string} url
 * @param {number} port
 * @param {boolean} is_tls
 * @param {string} login
 * @param {string} pass
 * @returns {PoolConfig}
 */
module.exports.pool_create = function(url, port, is_tls, login, pass) {
  /** @type {UnknownRecord} */
  const pool = {};
  for (const [key, definition] of Object.entries(poolTemplate)) {
    pool[key.replace(/^_/, "")] = cloneDefault(definition[0]);
  }
  Object.assign(pool, {url, port, is_tls, login, pass}, hostPoolDefaults(url));
  if (!isDefaultPool(pool)) {throw new Error("Default pool is malformed");}
  return pool;
};

// The MO donation endpoint: the multi-algo MoM proxy, which the host defaults put on the XMR
// `login` dialect. mom.js re-applies it before mining, so this is also what a config's donation
// pool is reset to.
/** @returns {PoolConfig} */
module.exports.donation_pool_create = function() {
  return module.exports.pool_create(momMultiAlgoProxyHost, 20001, true, "user", "mom");
};

const dev_help = 'device config "[<dev>[*MAIN|*[name=value;...]][^P],]+", dev = ' +
                 "{cpu, gpu<N>, cpu<N>}, N = device number, MAIN = algorithm primary tuning value, " +
                 "P = number of parallel processes";

module.exports.opt_help = {
  job: {
    _help:        "JSON string of the default job params (mostly used in test/bench mode)",
    algo:         [ null,  'algo name of the job (only used with "mine" directive)' ],
    dev:          [ "cpu", dev_help + "; in mine auto mode, restrict switching to these devices" ],
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
    _template: poolTemplate,
    _array: [module.exports.donation_pool_create()]
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
function isPlainObject(value) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {return false;}
  const prototype = Object.getPrototypeOf(value);
  return prototype === null || Object.getPrototypeOf(prototype) === null;
}

/** @param {unknown} value @returns {value is OptionSchema} */
function isOptionSchema(value) {
  return isPlainObject(value) &&
    (value["_help"] === undefined || typeof value["_help"] === "string") &&
    (value["_template"] === undefined || isPlainObject(value["_template"])) &&
    (value["_array"] === undefined ||
      (Array.isArray(value["_array"]) && value["_array"].every(isPlainObject))) &&
    (value["_map"] === undefined || isPlainObject(value["_map"]));
}

/** @param {unknown} value @param {string} key @returns {OptionSchema | null} */
function nestedOptionSchema(value, key) {
  if (!isPlainObject(value)) {return null;}
  if (!isOptionSchema(value)) {throw new Error(`Invalid option schema: ${key}`);}
  return value;
}

/** @param {unknown} value @returns {value is UnknownRecord[]} */
function isPlainObjectArray(value) {
  return Array.isArray(value) && value.every(isPlainObject);
}
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
const poolProtocols = new Set(["login", ...POOL_PROTOCOL_BY_ALGO.values()]);
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
// Each template's validator receives the raw JSON and its merge with the template defaults.
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
  if (!isPlainObject(val)) {
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

// Validate and normalize a merged pool template in place; returns an error text or null.
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
  try {pool["tls_fingerprint"] = normalizeTlsFingerprint(pool["tls_fingerprint"]);} catch (error) {
    return errorMessage(error);
  }
  if (pool["tls_fingerprint"] && !pool["is_tls"]) {return "tls_fingerprint requires is_tls";}
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

/** @param {unknown} defaultValue */
function formatDefaultValue(defaultValue) {
  switch (typeof defaultValue) {
    case "string":
      return "\"" + defaultValue + "\"";
    case "object":
      return JSON.stringify(defaultValue);
    default:
      return defaultValue;
  }
}

/** @param {unknown} defaultValue */
function defaultSuffix(defaultValue) {
  // null marks a public option with no default value (internal options are "_"-prefixed).
  return defaultValue !== undefined && defaultValue !== null ?
    " (" + formatDefaultValue(defaultValue) + " by default)" : "";
}

// Parse an option's JSON value into a plain object with string-typed job hex fields.
/** @param {string} arg @param {string} val @returns {UnknownRecord} */
function parseJsonObject(arg, val) {
  let parsed;
  try {
    parsed = JSON.parse(val);
  } catch (err) {
    const message = errorMessage(err);
    return module.exports.print_help("Can't parse option " + arg + " JSON param: " + message);
  }
  if (!isPlainObject(parsed)) {
    return optionError(arg, "JSON param must be an object");
  }
  const algo = parsed["algo"];
  if (algo !== undefined && algo !== null && typeof algo !== "string") {
    return optionError(arg, "has invalid algo value");
  }
  for (const field of ["blob_hex", "seed_hex", "target", "nicehash_mask"]) {
    if (Object.hasOwn(parsed, field) && typeof parsed[field] !== "string") {
      return optionError(arg, "has invalid " + field + " value");
    }
  }
  return parsed;
}

/** @param {string} arg @param {unknown} val */
function parseNonNegativeNumber(arg, val) {
  const num = numberValue(val);
  if (!Number.isFinite(num)) {
    return optionError(arg, "param must be a number: " + val);
  }
  if (num < 0) {
    return optionError(arg, "param must be non-negative: " + val);
  }
  return num;
}

/** @param {string} arg @param {unknown} val @param {string} keyPath */
function parseOptionNumber(arg, val, keyPath) {
  const num = parseNonNegativeNumber(arg, val);
  if (positiveTimerOptions.has(keyPath) && num <= 0) {
    return optionError(arg, "param must be positive: " + val);
  }
  if (timerOptions.has(keyPath) && num > maxTimerSeconds) {
    return optionError(arg, "param exceeds the maximum timer delay: " + val);
  }
  return num;
}

/** @param {string} arg @param {unknown} val */
function parseNonNegativeInteger(arg, val) {
  const num = parseNonNegativeNumber(arg, val);
  if (!Number.isSafeInteger(num)) {
    return optionError(arg, "param must be a safe integer: " + val);
  }
  return num;
}

/** @param {string} arg @param {unknown} val */
function parsePearlHashCertVersion(arg, val) {
  const version = parseNonNegativeInteger(arg, val);
  if (version !== 3) {
    return optionError(arg, "only supports PearlHash certificate version 3: " + val);
  }
  return version;
}

/** @param {string} arg @param {unknown} value @param {string} algo */
function parseDeviceRequest(arg, value, algo) {
  try {
    gpuTuning.parseDeviceList(value, algo);
  } catch (error) {
    const message = errorMessage(error);
    return optionError(arg, "has invalid dev value: " + value + " (" + message + ")");
  }
  return value;
}

/** @param {string} arg @param {unknown} value */
function parseBackend(arg, value) {
  try {
    return compilerPolicy.validateBackend(value);
  } catch (error) {
    const message = errorMessage(error);
    return optionError(arg, message);
  }
}

/**
 * Validates one job field; `--job '{...}'` and dotted `--job.<field>` both come through here so
 * neither form can skip a check. Dev tuning fields are checked against the normalized algorithm
 * from the same option; a dev given alone gets only the shared grammar check, because the
 * algorithm may come from a later option and resolution checks it per algorithm.
 * @param {string} arg @param {string} field @param {unknown} value @param {unknown} algo
 */
function parseJobField(arg, field, value, algo) {
  if (field === "dev") {
    return parseDeviceRequest(arg, value, normalizeAlgoName(typeof algo === "string" ? algo : null) || "");
  }
  if (field === "backend") {return parseBackend(arg, value);}
  if (field === "pearlhash_cert_version") {return parsePearlHashCertVersion(arg, value);}
  return jobIntegerFields.has(field) ? parseNonNegativeInteger(arg, value) : value;
}

/** @param {string} arg @param {unknown} val @param {number} max */
function parseBoundedInteger(arg, val, max) {
  const num = parseNonNegativeInteger(arg, val);
  if (num > max) {
    return optionError(arg, "param must be at most " + max + ": " + val);
  }
  return num;
}

/** @param {unknown} help */
function isNumberOption(help) {
  return Array.isArray(help) && typeof help[0] === "number";
}

// The JSON forms of an object option: --<key> for every object, plus --add.<key> for an array
// template and --new.<key>.<name> for a map template.
/** @param {string} arg @param {OptionSchema} keyHelp @param {string} keyPath
 * @param {string} newPrefix */
function isJsonOptionArg(arg, keyHelp, keyPath, newPrefix) {
  return arg === "--" + keyPath ||
         ("_array" in keyHelp && arg === "--add." + keyPath) ||
         ("_map" in keyHelp && arg.startsWith(newPrefix));
}

// Merge a template option's JSON over the template defaults; unknown fields and missing required
// fields (default undefined) are usage errors.
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
    return optionError(arg, `has unsupported field: ${unknown}`);
  }
  /** @type {UnknownRecord} */
  const result = {};
  for (const key2 of publicKeys(template)) {
    const definition = template[key2];
    if (!Array.isArray(definition)) {
      return optionError(arg, `has invalid template field: ${key2}`);
    }
    const defaultValue = definition[0];
    // template keys without a default (undefined) are required, so they must be supplied
    if (typeof defaultValue === "undefined" && !Object.hasOwn(values, key2)) {
      return module.exports.print_help("Need to specify key value \"" + key2 + "\" in " + key + " JSON");
    }
    result[key2] = Object.hasOwn(values, key2) ? values[key2] : defaultValue;
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
    return optionError(arg, "has invalid pool " + err);
  }
  for (const [field, fallback] of Object.entries(hostPoolDefaults(value["url"]))) {
    if (!Object.hasOwn(parsed, field)) {value[field] = fallback;}
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
    return optionError(arg, "has invalid algorithm name");
  }
  return normalized;
}

/** @param {string} arg @param {UnknownRecord} parsed @param {UnknownRecord} value */
function validateDefaultMsrOption(arg, parsed, value) {
  const register = mapNameFromOption(arg, "default_msr");
  if (!/^0x[0-9a-f]{1,8}$/i.test(register)) {
    return optionError(arg, "has invalid MSR register");
  }
  for (const field of ["value", "mask"]) {
    if (typeof value[field] !== "string" || !/^0x[0-9a-f]{1,16}$/i.test(value[field])) {
      return optionError(arg, `has invalid MSR ${field}: ${parsed[field]}`);
    }
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
    const message = errorMessage(error);
    return optionError(arg, "has invalid GPU tuning: " + message);
  }
  if (value["perf"] !== null) {
    const perf = numberValue(value["perf"]);
    if (!Number.isFinite(perf) || perf < 0) {
      return optionError(arg, "has invalid perf value: " + parsed["perf"]);
    }
    value["perf"] = perf;
  }
  value["backend"] = parseBackend(arg, value["backend"]);
  return value;
}

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {OptionSchema} keyHelp
 * @param {string} keyPath
 * @param {string} arg
 * @param {UnknownRecord} val
 */
function addArrayTemplateOption(opt, key, keyHelp, keyPath, arg, val) {
  if (!("_array" in keyHelp) || arg !== "--add." + keyPath) {
    return false;
  }
  const values = opt[key + "s"];
  if (!Array.isArray(values)) {return optionError(arg, "target is not an array");}
  values.push(val);
  return true;
}

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {OptionSchema} keyHelp
 * @param {string} newPrefix
 * @param {string} arg
 * @param {UnknownRecord} val
 */
function addMapTemplateOption(opt, key, keyHelp, newPrefix, arg, val) {
  if (!("_map" in keyHelp) || !arg.startsWith(newPrefix)) {
    return false;
  }
  const key2 = mapNameFromOption(arg, key);
  if (!key2 || isReservedKey(key2)) {
    return module.exports.print_help("Invalid option map key: " + key2);
  }
  const values = opt[key + "s"];
  if (!isPlainObject(values)) {return optionError(arg, "target is not a map");}
  values[key2] = val;
  return true;
}

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {OptionSchema} keyHelp
 * @param {string} keyPath
 * @param {string} newPrefix
 * @param {string} arg
 * @param {UnknownRecord} parsed
 */
function applyTemplateOption(opt, key, keyHelp, keyPath, newPrefix, arg, parsed) {
  if (!keyHelp._template) {return optionError(arg, "template is missing");}
  const defaults = templateDefaults(arg, key, keyHelp._template, parsed);
  const validator = templateValidators[keyPath];
  const value = validator ? validator(arg, parsed, defaults) : defaults;
  return addArrayTemplateOption(opt, key, keyHelp, keyPath, arg, value) ||
         addMapTemplateOption(opt, key, keyHelp, newPrefix, arg, value);
}

// Apply a plain JSON object option field by field. Job fields share the dotted --job.<field>
// validation; --job's dev is also kept as dev_request so automatic mode can save it.
/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {OptionSchema} keyHelp
 * @param {string} arg
 * @param {UnknownRecord} parsed
 */
function applyJsonOption(opt, key, keyHelp, arg, parsed) {
  const target = opt[key];
  if (!isPlainObject(target)) {return optionError(arg, "target is not an object");}
  for (const key2 of Object.keys(parsed)) {
    const supported = key === "job" ? jobJsonFields.has(key2) :
      isPublicKey(key2) && Object.hasOwn(keyHelp, key2);
    if (!supported) {
      return optionError(arg, `has unsupported field: ${key2}`);
    }
    let value = parsed[key2];
    if (key === "job") {
      value = parseJobField(arg + "." + key2, key2, value, parsed["algo"]);
    } else if (isNumberOption(keyHelp[key2])) {
      value = parseOptionNumber(arg + "." + key2, value, key + "." + key2);
    }
    target[key2] = value;
  }
  if (key === "job" && Object.hasOwn(parsed, "dev")) {
    target["dev_request"] = parsed["dev"];
  }
  return true;
}

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {OptionSchema} keyHelp
 * @param {string} keyPath
 * @param {string} newPrefix
 * @param {string} arg
 * @param {string} val
 */
function parseJsonOption(opt, key, keyHelp, keyPath, newPrefix, arg, val) {
  const parsed = parseJsonObject(arg, val);
  if (keyHelp._template) {
    return applyTemplateOption(opt, key, keyHelp, keyPath, newPrefix, arg, parsed);
  }
  return applyJsonOption(opt, key, keyHelp, arg, parsed);
}

// set opt object based on default values from opt_help object
/** @param {object} opt @param {OptionSchema} opt_help */
module.exports.set_default_opts = function(opt, opt_help) {
  if (!isPlainObject(opt)) {throw new Error("Default options target must be a plain object");}
  for (const key of publicKeys(opt_help)) {
    const keyHelp = opt_help[key];
    const schema = nestedOptionSchema(keyHelp, key);
    if (schema) {
      setObjectDefault(opt, key, schema);
    } else {
      opt[key] = simpleDefault(keyHelp);
    }
  }
};

/** @param {unknown} value @returns {value is PoolConfig} */
function isDefaultPool(value) {
  return isPlainObject(value) && typeof value["url"] === "string" &&
    typeof value["port"] === "number" && typeof value["is_tls"] === "boolean" &&
    typeof value["login"] === "string" && typeof value["pass"] === "string" &&
    typeof value["is_keepalive"] === "boolean" && typeof value["is_nicehash"] === "boolean" &&
    typeof value["use_subscribe"] === "boolean" && typeof value["tls_verify"] === "boolean" &&
    typeof value["tls_fingerprint"] === "string" &&
    (value["pearlhash_target_format"] === "base" || value["pearlhash_target_format"] === "jackpot") &&
    typeof value["worker"] === "string" && value["socket"] === null && value["keepalive"] === null &&
    value["last_job"] === null && isPlainObject(value["job_state"]) && typeof value["last_connect_time"] === "number" &&
    typeof value["pending_subscribe"] === "boolean" &&
    typeof value["pending_authorize"] === "boolean" &&
    typeof value["pending_submit_count"] === "number" &&
    Number.isSafeInteger(value["pending_submit_count"]) && value["pending_submit_count"] >= 0 &&
    typeof value["logged_in"] === "boolean" &&
    typeof value["good_shares"] === "number" && typeof value["bad_shares"] === "number";
}

// Check that schema-generated defaults produced every field MinerOptions promises.
/** @param {unknown} value @returns {value is MinerOptions} */
function isDefaultMinerOptions(value) {
  if (!isPlainObject(value) || !isPlainObject(value["job"]) || !isPlainObject(value["pool_time"]) ||
      !isPlainObject(value["pool_ids"]) || !isPlainObject(value["algo_params"]) ||
      !isPlainObject(value["default_msrs"]) || !Array.isArray(value["pools"]) ||
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
  if (!isPlainObject(poolIds)) {throw new Error("Default pool IDs are malformed");}
  poolIds["active"] = 0;
  if (!isDefaultMinerOptions(options)) {throw new Error("Default options are malformed");}
  return options;
};

/** @param {unknown} keyHelp */
function simpleDefault(keyHelp) {
  return Array.isArray(keyHelp) ? cloneDefault(keyHelp[0]) : keyHelp;
}

/** @param {UnknownRecord} opt @param {string} key @param {OptionSchema} keyHelp */
function setObjectDefault(opt, key, keyHelp) {
  // _array/_map templates seed the pluralized collection; plain objects recurse
  if ("_array" in keyHelp) {
    opt[key + "s"] = cloneDefault(keyHelp._array);
  } else if ("_map" in keyHelp) {
    opt[key + "s"] = cloneDefault(keyHelp._map);
  } else {
    module.exports.set_default_opts(opt[key] = {}, keyHelp);
  }
}

// The durable config: public options without the transient job, runtime pool state, or active id.
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
    // Keep automatic-mode selection durable without saving transient jobs or changing fixed mode.
    ...(!opt.job.algo && opt.job.dev_request !== undefined ? {job: {dev: opt.job.dev_request}} : {}),
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

/** @param {string} arg @param {string} message @returns {never} */
function optionError(arg, message) {
  return module.exports.print_help("Option " + arg + " " + message);
}

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
  if (!isPlainObject(values)) {
    return configError(key + " must be an object");
  }
  opt[key] = {};
  for (const [name, value] of Object.entries(values)) {
    if (!name || isReservedKey(name)) {
      return configError("invalid " + key + " key: " + name);
    }
    applyConfigOption(opt, "--new." + option + "." + name, value);
  }
}

/** @param {unknown} value @returns {UnknownRecord} */
function durablePoolFields(value) {
  if (!isPlainObject(value)) {
    return configError("pool must be an object");
  }
  return Object.fromEntries(Object.entries(value).filter(([key]) =>
    isPublicKey(key) && Object.hasOwn(module.exports.opt_help.pool._template, key)));
}

/** Retain unknown fields so the option parser reports typos; discard only legacy runtime state.
 * @param {unknown} value @returns {UnknownRecord} */
function configPoolFields(value) {
  if (!isPlainObject(value)) {
    return configError("pool must be an object");
  }
  return Object.fromEntries(Object.entries(value).filter(([key]) => !poolRuntimeFields.has(key)));
}

/** @param {MinerOptions} opt @param {unknown} value */
function applyPoolIds(opt, value) {
  if (!isPlainObject(value)) {
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
    if (!isSafeIntegerAtLeast(id, 0)) {
      return configError("pool_ids." + key + " must be a non-negative integer");
    }
    if (key === "primary") {opt.pool_ids.primary = id;}
    if (key === "donate") {opt.pool_ids.donate = id;}
  }
}

/** @param {MinerOptions} opt */
function validateConfigPoolIds(opt) {
  const validId = (/** @type {unknown} */ id) => isSafeIntegerAtLeast(id, 0) && id < opt.pools.length;
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

// Apply a parsed config file through the same option parser as the command line. Configured pools
// and maps replace their defaults; the result must name valid, distinct primary and donation pools.
/** @param {MinerOptions} opt @param {unknown} values */
module.exports.apply_config = function(opt, values) {
  if (!isPlainObject(values)) {
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
        if (isPlainObject(value) || Array.isArray(value) || value === null) {
          return configError(key + " must be a scalar");
        }
        applyConfigOption(opt, "--" + key, value);
        break;
      default: return configError("unsupported field: " + key);
    }
  }
  // Configured pools replace the defaults, including the donation pool at index 0. Unless the
  // config names a donation pool (an index or null), keep the default one after the user's pools
  // rather than letting pool_ids.donate=0 silently turn the user's first pool into it.
  if (Object.hasOwn(values, "pools") &&
      !(isPlainObject(values["pool_ids"]) && Object.hasOwn(values["pool_ids"], "donate"))) {
    opt.pools.push(module.exports.donation_pool_create());
    opt.pool_ids.donate = opt.pools.length - 1;
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
    const message = errorMessage(error);
    return module.exports.print_help("Can't load config file " + filename + ": " + message);
  }
  module.exports.apply_config(opt, values);
};

/** @param {string} file */
module.exports.is_config_file = function(file) {
  return path.extname(file).toLowerCase() === ".json";
};

// prints options help from opt_help
/** @param {OptionSchema} opt_help @param {string} indent @param {string} baseKeyPath */
function printOptHelp(opt_help, indent, baseKeyPath) {
  for (const key of publicKeys(opt_help)) {
    const keyPath = baseKeyPath ? baseKeyPath + "." + key : key;
    const definition = opt_help[key];
    const schema = nestedOptionSchema(definition, keyPath);
    if (schema) {
      printObjectOptHelp(schema, indent, keyPath);
    } else {
      printSimpleOptHelp(definition, indent, keyPath);
    }
  }
}

/** @param {string} line @param {string} help */
function paddedHelp(line, help) {
  return line.padEnd(36, " ") + help;
}

/** @param {OptionSchema} keyHelp @param {string} indent @param {string} keyPath */
function printTemplateHeader(keyHelp, indent, keyPath) {
  const fields = " '{[\"<key>\": <value>,]+}': ";
  if ("_array" in keyHelp) {
    console.log(paddedHelp(indent + "--add." + keyPath + fields, keyHelp._help || ""));
  } else if ("_map" in keyHelp) {
    console.log(paddedHelp(
      indent + "--new." + keyPath + ".<name>" + fields, keyHelp._help || ""));
  }
}

/** @param {UnknownRecord} template @param {string} indent */
function printTemplateFields(template, indent) {
  for (const key of publicKeys(template)) {
    const definition = template[key];
    if (!Array.isArray(definition) || typeof definition[1] !== "string") {
      throw new Error(`Invalid option help template field: ${key}`);
    }
    console.log(paddedHelp(indent + "  " + key + ": ",
      definition[1] + defaultSuffix(definition[0])));
  }
}

/** @param {OptionSchema} keyHelp @param {string} indent @param {string} keyPath */
function printObjectOptHelp(keyHelp, indent, keyPath) {
  if (typeof keyHelp._help === "undefined") {
    return;
  }
  if ("_template" in keyHelp) {
    printTemplateHeader(keyHelp, indent, keyPath);
    printTemplateFields(keyHelp._template, indent);
  } else {
    console.log(paddedHelp(indent + "--" + keyPath + " '{...}': ", keyHelp._help));
    printOptHelp(keyHelp, indent + "  ", keyPath);
  }
  console.log();
}

/** @param {unknown} keyHelp @param {string} indent @param {string} keyPath */
function printSimpleOptHelp(keyHelp, indent, keyPath) {
  if (!Array.isArray(keyHelp) || typeof keyHelp[1] !== "string") {
    throw new Error(`Invalid option help field: ${keyPath}`);
  }
  console.log(paddedHelp(indent + "--" + keyPath + ": ", keyHelp[1] + defaultSuffix(keyHelp[0])));
}

function helpCommand() {
  if (process.env["MOM_COMMAND"]) {
    return process.env["MOM_COMMAND"];
  }
  const exe = path.basename(process.argv[1] || "");
  return releaseCommandNames.has(exe) ? "./" + exe : "node mom.js";
}

/** @param {string | undefined} errorText @returns {never} */
function finishHelp(errorText) {
  if (errorText) {
    h.log_err(errorText);
  }
  process.exit(errorText ? 1 : 0);
}

/** @param {string} [errorText] @returns {never} */
module.exports.print_help = function(errorText) {
  const str = `
# MoM CPU/GPU miner v${VERSION}
Copyright (c) 2023-2026 MoneroOcean. GNU GPLv3.
$ ${helpCommand()} <directive> <parameter>+ [<option>+]

Directives:
  mine  (<pool_address:port[tls]> <login> [<pass>]|<config.json>)
  test  <algo> <result_hash_hex_str>
  bench <algo>
  algos       List supported algorithms and automatic device settings

Mining TLS option:
  --tls-fingerprint <SHA256>  Pin the primary pool certificate (requires a TLS port).
  Self-signed certificates are accepted by default; tls_verify enables CA/hostname checks.

Options:`;
  console.log(str);
  printOptHelp(this.opt_help, "", "");
  finishHelp(errorText);
};

// recursively parses all options specified by the opt_help data structure
/**
 * @param {object} opt
 * @param {OptionSchema} opt_help
 * @param {string} arg
 * @param {string} val
 * @param {string} baseKeyPath
 */
module.exports.parse_opt = function(opt, opt_help, arg, val, baseKeyPath) {
  if (!isPlainObject(opt)) {throw new Error("Option target must be a plain object");}
  for (const key of publicKeys(opt_help)) {
    const keyHelp = opt_help[key];
    const keyPath = baseKeyPath ? baseKeyPath + "." + key : key;
    if (parseOptionEntry(opt, key, keyHelp, keyPath, arg, val)) {
      return true;
    }
  }
  return false;
};

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {OptionSchema} keyHelp
 * @param {string} keyPath
 * @param {string} arg
 * @param {string} val
 */
function parseObjectOption(opt, key, keyHelp, keyPath, arg, val) {
  if (!("_help" in keyHelp)) {
    return false;
  }
  const newPrefix = "--new." + keyPath + ".";
  // consider val as JSON string here
  if (isJsonOptionArg(arg, keyHelp, keyPath, newPrefix)) {
    return parseJsonOption(opt, key, keyHelp, keyPath, newPrefix, arg, val);
  }
  const target = opt[key];
  if (!isPlainObject(target)) {return false;}
  // Dotted job fields take the JSON form's path so both forms share one set of job checks.
  const fieldPrefix = "--" + keyPath + ".";
  const field = arg.slice(fieldPrefix.length);
  if (keyPath === "job" && arg.startsWith(fieldPrefix) && isPublicKey(field) &&
      Object.hasOwn(keyHelp, field)) {
    return applyJsonOption(opt, key, keyHelp, "--job", {[field]: val});
  }
  return module.exports.parse_opt(target, keyHelp, arg, val, keyPath);
}

// Parse a scalar --<path> option; numeric options are range-checked, others are kept as strings.
/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {unknown} keyHelp
 * @param {string} keyPath
 * @param {string} arg
 * @param {string} val
 */
function parseSimpleOption(opt, key, keyHelp, keyPath, arg, val) {
  if (arg !== "--" + keyPath) {
    return false;
  }
  const max = boundedIntegerOptions.get(keyPath);
  opt[key] = max === undefined ?
    (isNumberOption(keyHelp) ? parseOptionNumber(arg, val, keyPath) : val) :
    parseBoundedInteger(arg, val, max);
  return true;
}

/**
 * @param {UnknownRecord} opt
 * @param {string} key
 * @param {unknown} keyHelp
 * @param {string} keyPath
 * @param {string} arg
 * @param {string} val
 */
function parseOptionEntry(opt, key, keyHelp, keyPath, arg, val) {
  const schema = nestedOptionSchema(keyHelp, keyPath);
  if (schema) {
    return parseObjectOption(opt, key, schema, keyPath, arg, val);
  }
  return parseSimpleOption(opt, key, keyHelp, keyPath, arg, val);
}

// inject internal default values to opt object from opt_help object
/** @param {object} opt @param {OptionSchema} opt_help */
module.exports.set_internal_opts = function(opt, opt_help) {
  if (!isPlainObject(opt)) {throw new Error("Internal options target must be a plain object");}
  for (const key of publicKeys(opt_help)) {
    const definition = opt_help[key];
    const schema = nestedOptionSchema(definition, key);
    if (schema) {setInternalObject(opt, key, schema);}
  }
};

/** @param {UnknownRecord} opt @param {string} key @param {OptionSchema} keyHelp */
function templateItems(opt, key, keyHelp) {
  const collection = opt[key + "s"];
  let values;
  if ("_array" in keyHelp) {
    if (!Array.isArray(collection)) {
      throw new Error(`Invalid internal option collection: ${key}`);
    }
    values = collection;
  } else if ("_map" in keyHelp) {
    if (!isPlainObject(collection)) {
      throw new Error(`Invalid internal option collection: ${key}`);
    }
    values = Object.values(collection);
  } else {
    throw new Error(`Internal option template has no collection: ${key}`);
  }
  if (!isPlainObjectArray(values)) {
    throw new Error(`Invalid internal option collection: ${key}`);
  }
  return values;
}

/** @param {UnknownRecord} item @param {UnknownRecord} template */
function applyInternalTemplateValues(item, template) {
  for (const [key, val] of Object.entries(template)) {
    if (key.startsWith("_")) {
      if (!Array.isArray(val)) {throw new Error(`Invalid internal option field: ${key}`);}
      item[key.substring(1)] = cloneDefault(val[0]);
    }
  }
}

/** @param {UnknownRecord} opt @param {string} key @param {OptionSchema} keyHelp */
function setInternalObject(opt, key, keyHelp) {
  if (!keyHelp._template) {
    const target = opt[key];
    if (!isPlainObject(target)) {throw new Error(`Invalid internal option object: ${key}`);}
    return module.exports.set_internal_opts(target, keyHelp);
  }
  for (const item of templateItems(opt, key, keyHelp)) {
    applyInternalTemplateValues(item, keyHelp._template);
  }
}
