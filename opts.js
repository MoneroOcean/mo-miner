// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

"use strict";

const path = require("path");
const h    = require("./helper.js");

const version_str = require("./package.json").version;

module.exports.agent_str = "mo-miner v" + version_str;

module.exports.pool_create = function(url, port, is_tls, login, pass) {
  return {
    url:          url,
    port:         port,
    is_tls:       is_tls,
    is_nicehash:  url.includes("nicehash") ? true : false,
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
  save_config: [ "", "file name to save config in JSON format (only for mine directive)" ]
};

// object but not array
const isObject = function(a) { return (!!a) && (a.constructor === Object); }

// set opt object based on default values from opt_help object
module.exports.set_default_opts = function(opt, opt_help) {
  for (const key in opt_help) {
    if (key.startsWith("_")) continue;
    if (isObject(opt_help[key])) {
      if ("_array" in opt_help[key]) {
        opt[key + "s"] = opt_help[key]._array;
      } else if ("_map" in opt_help[key]) {
        opt[key + "s"] = opt_help[key]._map;
      } else {
        opt[key] = {};
        this.set_default_opts(opt[key], opt_help[key]);
      }
    } else if (Array.isArray(opt_help[key])) {
      opt[key] = opt_help[key][0];
    } else {
      opt[key] = opt_help[key];
    }
  }
};

module.exports.saved_config = function(opt) {
  const saved = { ...opt };
  delete saved.job;
  return saved;
};

// prints options help from opt_help
module.exports.print_opt_help = function(opt_help, depth_str, base_key_path_str) {
  const pad = 36;
  for (const key in opt_help) {
    if (key.startsWith("_")) continue;
    const key_path_str = base_key_path_str ? base_key_path_str + "." + key : key;
    if (isObject(opt_help[key])) {
      const key_help = opt_help[key]._help;
      if (typeof opt_help[key]._help === 'undefined') continue;
      if ("_template" in opt_help[key]) {
        if ("_array" in opt_help[key]) {
          console.log(( depth_str + "--add." + key_path_str +
                        " '{[\"<key>\": <value>,]+}': ").padEnd(pad, " ") + key_help);
        } else if ("_map" in opt_help[key]) {
          console.log(( depth_str + "--new." + key_path_str + ".<name>" +
                        " '{[\"<key>\": <value>,]+}': ").padEnd(pad, " ") + key_help);
        }
        const template = opt_help[key]._template;
        for (const key2 in template) {
          if (key2.startsWith("_")) continue;
          let def_val = template[key2][0];
          if (def_val === null) continue; // do not show internal params
          switch (typeof def_val) {
            case 'string': def_val = "\"" + def_val + "\""; break;
            case 'bigint': def_val = "0x" + def_val.toString(16); break;
          }
          console.log((depth_str + "  " + key2 + ": ").padEnd(pad, " ") + template[key2][1] +
                      (typeof def_val !== 'undefined' ? " (" + def_val + " by default)" : ""));
        }
      } else {
        console.log((depth_str + "--" + key_path_str + " '{...}': ").padEnd(pad, " ") + key_help);
        this.print_opt_help(opt_help[key], depth_str + "  ", key_path_str);
      }
      console.log();
    } else {
      let def_val = opt_help[key][0];
      if (def_val === null) continue; // do not show internal params
      switch (typeof def_val) {
        case 'string': def_val = "\"" + def_val + "\""; break;
        case 'bigint': def_val = "0x" + def_val.toString(16); break;
      }
      console.log((depth_str + "--" + key_path_str + ": ").padEnd(pad, " ") + opt_help[key][1] +
                  (typeof def_val !== 'undefined' ? " (" + def_val + " by default)" : ""));
    }
  }
};

module.exports.print_help = function(err_str) {
  const exe = path.basename(process.argv[1] || "");
  const command = process.env.MOMINER_COMMAND ||
                  (exe === "mo-miner" || exe === "mo-miner.exe" ? "./" + exe : "node mo-miner.js");
  const str = `
# Node.js/SYCL based CPU/GPU miner v${version_str}
$ ${command} <directive> <parameter>+ [<option>+]

Directives:
  mine  (<pool_address:port[tls]> <login> [<pass>]|<config.json>)
  test  <algo> <result_hash_hex_str>
  bench <algo>
  algo_params

Options:`;
  console.log(str);
  this.print_opt_help(this.opt_help, "", "");
  if (err_str) {
    h.log_err(err_str);
    process.exit(1);
  }
  process.exit(0);
};

// recursively parses all options specified by the opt_help data structure
module.exports.parse_opt = function(opt, opt_help, arg, val, base_key_path_str) {
  for (const key in opt_help) {
    if (key.startsWith("_")) continue;
    const key_path_str = base_key_path_str ? base_key_path_str + "." + key : key;
    if (isObject(opt_help[key])) {
      if (!("_help" in opt_help[key])) continue;
      const new_str_prefix = "--new." + key_path_str + ".";
      if (arg === "--" + key_path_str ||
          arg === "--add." + key_path_str ||
          arg.startsWith(new_str_prefix)
      ) {
        // consider val as JSON string here
        let val2;
        try {
          val2 = JSON.parse(val);
        } catch (err) {
          return this.print_help("Can't parse option " + arg + " JSON param: " + val + ": " + err);
        }
        if ("_template" in opt_help[key]) {
          const template = opt_help[key]._template;
          // val3 is final value including defaults from opt_help
          let val3 = {};
          for (const key2 in template) {
            if (key2.startsWith("_")) continue;
            const default_val = template[key2][0];
            const def_val = template[key2][0];
            // do not allow to keys without default values to be missing
            if (typeof def_val === 'undefined' && !(key2 in val2))
              return this.print_help("Need to specify key value \"" + key2 + "\" in " + key + " JSON");
            val3[key2] = key2 in val2 ? val2[key2] : def_val;
          }
          if ("_array" in opt_help[key] && arg === "--add." + key_path_str) {
            opt[key + "s"].push(val3);
            return true;
          } else if ("_map" in opt_help[key] && arg.startsWith(new_str_prefix)) {
            const key2 = arg.substring(new_str_prefix.length);
            opt[key + "s"][key2] = val3;
            return true;
          }
        } else {
          for (const key2 in val2) opt[key][key2] = val2[key2];
          return true;
        }
      } else if (this.parse_opt(opt[key], opt_help[key], arg, val, key_path_str)) return true;

    // sets simple key value
    } else if (arg === "--" + key_path_str) {
      opt[key] = typeof opt_help[key][0] == 'number' ? Number(val) : val;
      return true;
    }
  }
  return false;
};

// inject internal default values to opt object from opt_help object
module.exports.set_internal_opts = function(opt, opt_help) {
  for (const key in opt_help) {
    if (key.startsWith("_")) continue;
    if (isObject(opt_help[key])) {
      if ("_template" in opt_help[key]) {
        let items = "_array" in opt_help[key] ? opt[key + "s"] : Object.values(opt[key + "s"]);
        for (let item of items) {
          for (const [key2, val2] of Object.entries(opt_help[key]._template)) {
            if (!key2.startsWith("_")) continue;
            item[key2.substring(1)] = val2[0];
          }
        }
      } else this.set_internal_opts(opt[key], opt_help[key]);
    }
  }
};
