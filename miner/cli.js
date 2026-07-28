"use strict";

module.exports = ({h, o, opt, path, normalizeAlgoName}) => {
  function mergeNestedConfigOption(key, values) {
    for (const nestedKey in values) {opt[key][nestedKey] = values[nestedKey];}
  }

  function mergeConfigOptions(values) {
    for (const key in values) {
      switch (key) {
        case "job":
        case "pool_time":
          mergeNestedConfigOption(key, values[key]);
          break;
        default:
          opt[key] = values[key];
      }
    }
  }

  function loadConfigFile(configFile) {
    const filename = path.resolve(configFile);
    h.log("Loading config file " + filename);
    mergeConfigOptions(require(filename));
  }

  function parsePoolPort(value) {
    const match = value.match(/^(\d+)((?:tls)?)$/);
    if (!match) {return o.print_help("Wrong pool port: " + value);}
    const port = Number(match[1]);
    if (port < 1 || port > 65535) {return o.print_help("Wrong pool port: " + value);}
    return {port, is_tls: match[2] === "tls"};
  }

  function parsePoolUri(uri) {
    const parts = uri.split(":");
    if (parts.length !== 2) {return o.print_help("Wrong pool URI: " + uri);}
    const parsed = parsePoolPort(parts[1]);
    if (!parsed) {return parsed;}
    return {url: parts[0], port: parsed.port, is_tls: parsed.is_tls};
  }

  function addPrimaryPool(uri, login, pass) {
    const pool = parsePoolUri(uri);
    opt.pool_ids.primary = opt.pools.length;
    opt.pools.push(o.pool_create(pool.url, pool.port, pool.is_tls, login, pass));
  }

  function optionalPoolPass(args) {
    return args.length > 0 && !args[0].match(/^--/) ? args.shift() : "";
  }

  function parseMineArgs(args) {
    if (args.length < 1) {return o.print_help("Directive \"mine\" needs 1+ parameters");}
    const first = args.shift();
    if (o.is_config_file(first)) {return loadConfigFile(first);}
    if (args.length < 1) {return o.print_help("Directive \"mine\" needs 2+ parameters");}
    return addPrimaryPool(first, args.shift(), optionalPoolPass(args));
  }

  function parseTestArgs(args) {
    if (args.length < 2) {return o.print_help("Directive \"test\" needs two parameters");}
    opt.job.algo = normalizeAlgoName(args.shift());
    return args.shift();
  }

  function parseBenchArgs(args) {
    if (args.length < 1) {return o.print_help("Directive \"bench\" needs one parameter");}
    opt.job.algo = normalizeAlgoName(args.shift());
  }

  function parseRemainingOptions(args) {
    while (args.length) {
      const arg = args.shift();
      if (args.length >= 1 && o.parse_opt(opt, o.opt_help, arg, args[0], "")) {args.shift();}
      else {return o.print_help("Unparsed option: " + arg);}
    }
  }

  return function parseArgs(argv, test) {
    const args = argv.slice(2);
    if (args.length === 0) {return o.print_help("No directive specified");}
    const directive = args.shift();
    const parsers = {
      mine: parseMineArgs,
      test: (remaining) => { test.result_hash_hex = parseTestArgs(remaining); },
      bench: parseBenchArgs,
      algo_params: () => undefined,
    };
    const parser = parsers[directive];
    if (!parser) {return o.print_help("Unknown directive " + directive);}
    parser(args);
    parseRemainingOptions(args);
    opt.job.algo = normalizeAlgoName(opt.job.algo);
    return directive;
  };
};
