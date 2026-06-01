// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

"use strict";

const path = require("path");
const h    = require("./helper.js");

const version_str = require("./package.json").version;

module.exports.agent_str = "mo-miner v" + version_str;
const releaseCommandNames = new Set(["mo-miner", "mo-miner.exe"]);

module.exports.pool_create = function(url, port, is_tls, login, pass) {
  return {
    url:          url,
    port:         port,
    is_tls:       is_tls,
    tls_verify:   false,
    is_nicehash:  url.includes("nicehash"),
    is_keepalive: true,
    login:        login,
    pass:         pass
  };
};

const dev_help = 'device config line "[<dev>[*B][^P],]+", dev = ' +
                 '{cpu, gpu<N>, cpu<N>}, ' +
                 'N = device number, B = hash batch size, P = number of parallel processes';

module.exports.opt_help = {
  job: {
    _help:        'JSON string of the default job params (mostly used in test/bench mode)',
    algo:         [ null,  'algo name of the job (only used with "mine" directive)' ],
    dev:          [ "cpu", dev_help ],
    blob_hex: [ "0305A0DBD6BF05CF16E503F3A66F78007CBF34144332ECBFC22ED95C8700383B309ACE1923A0964B"
              + "00000008BA939A62724C0D7581FCE5761E9D8A0E6A1C3F924FDD8493D1115649C05EB601",
                'hexadecimal string of input blob' ],
    seed_hex: [ "3132333435363738393031323334353637383930313233343536373839303132",
                'hexadecimal string of seed hash blob (used for rx algos)' ],
    height:   [ 0, "Block height used by some algos"],
  },
  pool_time: {
    _help:             'JSON string of pool related timings (in seconds)',
    stats:             [ 10*60,  'time to show pool mining stats' ],
    connect_throttle:  [ 60,     'time between pool connection attempts' ],
    primary_reconnect: [ 90,     'time to try to use primary pool if currently on backup pool' ],
    first_job_wait:    [ 15,     'consider pool bad if no first job after connection' ],
    close_wait:        [ 10,     'keep pool socket to submit delayed jobs' ],
    donate_interval:   [ 100*60, 'time before donation pool is activated' ],
    donate_length:     [ 1*60,   'donation pool work time' ],
    keepalive:         [ 5*60,   'interval to send keepalive messages' ],
  },
  pool: {
    _help: 'add backup pool, defined by the following keys:',
    _template: {
      url:                [ undefined, "pool DNS or IP address" ],
      port:               [ undefined, "pool port" ],
      is_tls:             [ false, "is pool port is encrypted using TLS/SSL" ],
      tls_verify:         [ false, "verify pool TLS/SSL certificate" ],
      is_nicehash:        [ false, "nicehash nonce mining mode support" ],
      is_keepalive:       [ true, "sends keepalive messages to the pool to avoid disconnect" ],
      login:              [ undefined, "pool login data" ],
      pass:               [ "", "pool password" ],
      _socket:            [ null, "network socket object" ],
      _keepalive:         [ null, "keepalive timer object" ],
      _last_connect_time: [ 0, "last connect time for throttling purposes" ],
      _last_job:          [ null, "last job object" ],
      _good_shares:       [ 0, "number of accepted shares" ],
      _bad_shares:        [ 0, "number of invalid shares" ],
    },
    _array: [
      this.pool_create("xmrig.moneroocean.stream", 20001, true, "user", "pass")
    ]
  },
  default_msr: {
    _help: 'stores default MSR register values to restore them without reboot, ' +
           'keys should be hex strings with 0x prefix',
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
    _help: 'new algo params, defined by the following keys:',
    _template: {
      dev:      [ "cpu", dev_help ],
      perf:     [ null, "algo hashrate" ],
    },
    _map: {}
  },
  log_level: [ 0, "log level: 0=minimal, 1=verbose, 2=network debug, 3=compute core debug" ],
  bench_algo_params: [ 1, "benchmark algo params before mining; set 0 to skip hashrate benchmarking" ],
  save_config: [ "", "file name to save config in JSON format (only for mine directive)" ]
};

// object but not array
const isObject = function(a) { return (!!a) && (a.constructor === Object); }
const poolBooleanFields = ["is_tls", "tls_verify", "is_nicehash", "is_keepalive"];
const templateValidators = {
  pool: validatePoolOption,
  algo_param: validateAlgoParamOption,
};

function cloneDefault(val) {
  if (Array.isArray(val)) return val.map(cloneDefault);
  if (!isObject(val)) return val;
  const cloned = {};
  for (const key in val) cloned[key] = cloneDefault(val[key]);
  return cloned;
}

function isPublicKey(key) {
  return !key.startsWith("_");
}

function publicKeys(object) {
  return Object.keys(object).filter(isPublicKey);
}

function isValidPort(port) {
  return Number.isInteger(port) && port >= 1 && port <= 65535;
}

function validatePoolUrl(pool) {
  return typeof pool.url === "string" && pool.url !== "" ? null : "url must be a non-empty string";
}

function validatePoolPort(pool) {
  const port = Number(pool.port);
  if (!isValidPort(port)) return "port must be an integer from 1 to 65535";
  pool.port = port;
  return null;
}

function validatePoolBooleans(pool) {
  const invalidBoolean = poolBooleanFields.find((key) => typeof pool[key] !== "boolean");
  return invalidBoolean ? invalidBoolean + " must be boolean" : null;
}

function validatePool(pool) {
  return validatePoolUrl(pool) || validatePoolPort(pool) || validatePoolBooleans(pool);
}

function formatDefaultValue(def_val) {
  switch (typeof def_val) {
    case 'string': return "\"" + def_val + "\"";
    case 'bigint': return "0x" + def_val.toString(16);
    default: return def_val;
  }
}

function defaultSuffix(def_val) {
  return typeof def_val !== 'undefined' ? " (" + formatDefaultValue(def_val) + " by default)" : "";
}

function parseJsonObject(arg, val) {
  let parsed;
  try {
    parsed = JSON.parse(val);
  } catch (err) {
    return module.exports.print_help("Can't parse option " + arg + " JSON param: " + val + ": " + err);
  }
  return validateParsedJsonObject(arg, val, parsed);
}

function validateParsedJsonObject(arg, val, parsed) {
  if (!isObject(parsed))
    return module.exports.print_help("Option " + arg + " JSON param must be an object: " + val);
  if ("dev" in parsed && !h.is_valid_dev(parsed.dev))
    return module.exports.print_help("Option " + arg + " has invalid dev value: " + parsed.dev);
  return parsed;
}

function parseNonNegativeNumber(arg, val) {
  const val2 = Number(val);
  if (!Number.isFinite(val2))
    return module.exports.print_help("Option " + arg + " param must be a number: " + val);
  if (val2 < 0)
    return module.exports.print_help("Option " + arg + " param must be non-negative: " + val);
  return val2;
}

function isNumberOption(help) {
  return Array.isArray(help) && typeof help[0] === 'number';
}

function isJsonOptionArg(arg, key_path_str, new_str_prefix) {
  return arg === "--" + key_path_str ||
         arg === "--add." + key_path_str ||
         arg.startsWith(new_str_prefix);
}

function templateDefaultValue(key2, def_val, values) {
  return key2 in values ? values[key2] : def_val;
}

function templateDefaults(arg, key, template, values) {
  const result = {};
  for (const key2 of publicKeys(template)) {
    const def_val = template[key2][0];
    // do not allow to keys without default values to be missing
    if (typeof def_val === 'undefined' && !(key2 in values))
      return module.exports.print_help("Need to specify key value \"" + key2 + "\" in " + key + " JSON");
    result[key2] = templateDefaultValue(key2, def_val, values);
  }
  return result;
}

function validatePoolOption(arg, val2, val3) {
  const err = validatePool(val3);
  return err ? module.exports.print_help("Option " + arg + " has invalid pool " + err) : val3;
}

function validateAlgoPerf(arg, val2, val3) {
  if (val3.perf === null) return val3;
  val3.perf = Number(val3.perf);
  if (!Number.isFinite(val3.perf) || val3.perf < 0) {
    return module.exports.print_help("Option " + arg + " has invalid perf value: " + val2.perf);
  }
  return val3;
}

function validateAlgoParamOption(arg, val2, val3) {
  if (!h.is_valid_dev(val3.dev))
    return module.exports.print_help("Option " + arg + " has invalid dev value: " + val3.dev);
  return validateAlgoPerf(arg, val2, val3);
}

function validateTemplateValue(arg, key_path_str, val2, val3) {
  const validator = templateValidators[key_path_str];
  return validator ? validator(arg, val2, val3) : val3;
}

function addArrayTemplateOption(opt, key, key_help, key_path_str, arg, val) {
  if (!("_array" in key_help) || arg !== "--add." + key_path_str) return false;
  opt[key + "s"].push(val);
  return true;
}

function addMapTemplateOption(opt, key, key_help, new_str_prefix, arg, val) {
  if (!("_map" in key_help) || !arg.startsWith(new_str_prefix)) return false;
  const key2 = arg.substring(new_str_prefix.length);
  opt[key + "s"][key2] = val;
  return true;
}

function applyTemplateOption(opt, key, key_help, key_path_str, new_str_prefix, arg, val2) {
  const template = key_help._template;
  // val3 is final value including defaults from opt_help
  const val3 = validateTemplateValue(arg, key_path_str, val2, templateDefaults(arg, key, template, val2));
  return addArrayTemplateOption(opt, key, key_help, key_path_str, arg, val3) ||
         addMapTemplateOption(opt, key, key_help, new_str_prefix, arg, val3);
}

function applyJsonOption(opt, key, key_help, arg, val2) {
  for (const key2 in val2) {
    const help = key_help[key2];
    opt[key][key2] = isNumberOption(help) ? parseNonNegativeNumber(arg + "." + key2, val2[key2]) : val2[key2];
  }
  return true;
}

function parseJsonOption(opt, key, key_help, key_path_str, new_str_prefix, arg, val) {
  const val2 = parseJsonObject(arg, val);
  if ("_template" in key_help)
    return applyTemplateOption(opt, key, key_help, key_path_str, new_str_prefix, arg, val2);
  return applyJsonOption(opt, key, key_help, arg, val2);
}

// set opt object based on default values from opt_help object
module.exports.set_default_opts = function(opt, opt_help) {
  for (const key of publicKeys(opt_help)) {
    const key_help = opt_help[key];
    if (isObject(key_help)) setObjectDefault(opt, key, key_help);
    else opt[key] = simpleDefault(key_help);
  }
};

function simpleDefault(key_help) {
  return Array.isArray(key_help) ? cloneDefault(key_help[0]) : key_help;
}

function setObjectDefault(opt, key, key_help) {
  if ("_array" in key_help) {
    opt[key + "s"] = cloneDefault(key_help._array);
  } else if ("_map" in key_help) {
    opt[key + "s"] = cloneDefault(key_help._map);
  } else {
    opt[key] = {};
    module.exports.set_default_opts(opt[key], key_help);
  }
}

module.exports.saved_config = function(opt) {
  const saved = { ...opt };
  delete saved.job;
  return saved;
};

module.exports.is_config_file = function(file) {
  return path.extname(file).toLowerCase() === ".json";
};

// prints options help from opt_help
module.exports.print_opt_help = function(opt_help, depth_str, base_key_path_str) {
  for (const key of publicKeys(opt_help)) {
    const key_path_str = base_key_path_str ? base_key_path_str + "." + key : key;
    if (isObject(opt_help[key])) printObjectOptHelp(opt_help[key], depth_str, key_path_str);
    else printSimpleOptHelp(opt_help[key], depth_str, key_path_str);
  }
};

function paddedHelp(line, help) {
  return line.padEnd(36, " ") + help;
}

function printTemplateHeader(key_help, depth_str, key_path_str) {
  if ("_array" in key_help)
    console.log(paddedHelp(depth_str + "--add." + key_path_str + " '{[\"<key>\": <value>,]+}': ", key_help._help));
  else if ("_map" in key_help)
    console.log(paddedHelp(depth_str + "--new." + key_path_str + ".<name> '{[\"<key>\": <value>,]+}': ", key_help._help));
}

function printTemplateFields(template, depth_str) {
  for (const key of publicKeys(template)) {
    const def_val = template[key][0];
    if (def_val === null) continue; // do not show internal params
    console.log(paddedHelp(depth_str + "  " + key + ": ", template[key][1] + defaultSuffix(def_val)));
  }
}

function printObjectOptHelp(key_help, depth_str, key_path_str) {
  if (typeof key_help._help === 'undefined') return;
  if ("_template" in key_help) {
    printTemplateHeader(key_help, depth_str, key_path_str);
    printTemplateFields(key_help._template, depth_str);
  } else {
    console.log(paddedHelp(depth_str + "--" + key_path_str + " '{...}': ", key_help._help));
    module.exports.print_opt_help(key_help, depth_str + "  ", key_path_str);
  }
  console.log();
}

function printSimpleOptHelp(key_help, depth_str, key_path_str) {
  const def_val = key_help[0];
  if (def_val === null) return; // do not show internal params
  console.log(paddedHelp(depth_str + "--" + key_path_str + ": ", key_help[1] + defaultSuffix(def_val)));
}

function helpCommand() {
  const exe = path.basename(process.argv[1] || "");
  if (process.env.MOMINER_COMMAND) return process.env.MOMINER_COMMAND;
  return releaseCommandNames.has(exe) ? "./" + exe : "node mo-miner.js";
}

function finishHelp(err_str) {
  if (err_str) {
    h.log_err(err_str);
    process.exit(1);
  }
  process.exit(0);
}

module.exports.print_help = function(err_str) {
  const str = `
# Node.js/SYCL based CPU/GPU miner v${version_str}
$ ${helpCommand()} <directive> <parameter>+ [<option>+]

Directives:
  mine  (<pool_address:port[tls]> <login> [<pass>]|<config.json>)
  test  <algo> <result_hash_hex_str>
  bench <algo>
  algo_params

Options:`;
  console.log(str);
  this.print_opt_help(this.opt_help, "", "");
  finishHelp(err_str);
};

// recursively parses all options specified by the opt_help data structure
module.exports.parse_opt = function(opt, opt_help, arg, val, base_key_path_str) {
  for (const key of publicKeys(opt_help)) {
    const key_help = opt_help[key];
    const key_path_str = base_key_path_str ? base_key_path_str + "." + key : key;
    if (parseOptionEntry(opt, key, key_help, key_path_str, arg, val)) return true;
  }
  return false;
};

function parseObjectOption(opt, key, key_help, key_path_str, arg, val) {
  if (!("_help" in key_help)) return false;
  const new_str_prefix = "--new." + key_path_str + ".";
  // consider val as JSON string here
  if (isJsonOptionArg(arg, key_path_str, new_str_prefix))
    return parseJsonOption(opt, key, key_help, key_path_str, new_str_prefix, arg, val);
  return module.exports.parse_opt(opt[key], key_help, arg, val, key_path_str);
}

function parseSimpleOption(opt, key, key_help, key_path_str, arg, val) {
  if (arg !== "--" + key_path_str) return false;
  opt[key] = isNumberOption(key_help) ? parseNonNegativeNumber(arg, val) : val;
  return true;
}

function parseOptionEntry(opt, key, key_help, key_path_str, arg, val) {
  if (isObject(key_help)) return parseObjectOption(opt, key, key_help, key_path_str, arg, val);
  return parseSimpleOption(opt, key, key_help, key_path_str, arg, val);
}

// inject internal default values to opt object from opt_help object
module.exports.set_internal_opts = function(opt, opt_help) {
  for (const key of publicKeys(opt_help)) {
    if (isObject(opt_help[key])) setInternalObject(opt, key, opt_help[key]);
  }
};

function templateItems(opt, key, key_help) {
  return "_array" in key_help ? opt[key + "s"] : Object.values(opt[key + "s"]);
}

function applyInternalTemplateValues(item, template) {
  for (const [key, val] of Object.entries(template)) {
    if (key.startsWith("_")) item[key.substring(1)] = val[0];
  }
}

function setInternalObject(opt, key, key_help) {
  if (!("_template" in key_help)) return module.exports.set_internal_opts(opt[key], key_help);
  for (let item of templateItems(opt, key, key_help)) {
    applyInternalTemplateValues(item, key_help._template);
  }
}
