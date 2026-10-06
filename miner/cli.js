"use strict";

const net = require("node:net");

/**
 * @param {{
 *   o: {
 *     is_config_file(file: string): boolean,
 *     load_config(opt: MinerOptions, file: string): unknown,
 *     opt_help: UnknownRecord,
 *     parse_opt(opt: object, help: UnknownRecord, arg: string, value: string, base: string): boolean,
 *     pool_create(url: string, port: number, tls: boolean, login: string, pass: string): PoolConfig,
 *     print_help(message?: string): never,
 *   },
 *   opt: MinerOptions,
 *   normalizeAlgoName(algo: string | null | undefined): string | null | undefined,
 * }} dependencies
 */
module.exports = ({o, opt, normalizeAlgoName}) => {
  /** @param {string} value */
  function parsePoolPort(value) {
    const match = value.match(/^(\d+)((?:tls)?)$/);
    if (!match) {return o.print_help("Wrong pool port: " + value);}
    const port = Number(match[1]);
    if (port < 1 || port > 65535) {return o.print_help("Wrong pool port: " + value);}
    return {port, is_tls: match[2] === "tls"};
  }

  /** @param {string} uri */
  function parsePoolUri(uri) {
    if (uri.startsWith("[")) {
      const match = uri.match(/^\[([^[]+)\]:(\d+(?:tls)?)$/);
      const host = match?.[1];
      const portText = match?.[2];
      if (!host || !portText || net.isIP(host) !== 6) {
        return o.print_help("Wrong pool URI: " + uri);
      }
      const parsed = parsePoolPort(portText);
      return {url: host, port: parsed.port, is_tls: parsed.is_tls};
    }
    const parts = uri.split(":");
    if (parts.length !== 2) {return o.print_help("Wrong pool URI: " + uri);}
    const url = parts[0];
    const port = parts[1];
    if (!url || port === undefined) {return o.print_help("Wrong pool URI: " + uri);}
    const parsed = parsePoolPort(port);
    return {url, port: parsed.port, is_tls: parsed.is_tls};
  }

  /** @param {string} uri @param {string} login @param {string} pass */
  function addPrimaryPool(uri, login, pass) {
    const pool = parsePoolUri(uri);
    opt.pool_ids.primary = opt.pools.length;
    opt.pools.push(o.pool_create(pool.url, pool.port, pool.is_tls, login, pass));
  }

  /** @param {string[]} args */
  function optionalPoolPass(args) {
    const first = args[0];
    return first !== undefined && !first.match(/^--/) ? args.shift() || "" : "";
  }

  /** @param {string[]} args */
  function parseMineArgs(args) {
    const first = args.shift();
    if (first === undefined) {return o.print_help("Directive \"mine\" needs 1+ parameters");}
    if (o.is_config_file(first)) {return o.load_config(opt, first);}
    const login = args.shift();
    if (login === undefined) {return o.print_help("Directive \"mine\" needs 2+ parameters");}
    return addPrimaryPool(first, login, optionalPoolPass(args));
  }

  /** @param {string[]} args */
  function parseTestArgs(args) {
    const algo = args.shift();
    const expected = args.shift();
    if (algo === undefined || expected === undefined) {
      return o.print_help("Directive \"test\" needs two parameters");
    }
    opt.job.algo = normalizeAlgoName(algo) || null;
    return expected;
  }

  /** @param {string[]} args */
  function parseBenchArgs(args) {
    const algo = args.shift();
    if (algo === undefined) {return o.print_help("Directive \"bench\" needs one parameter");}
    opt.job.algo = normalizeAlgoName(algo) || null;
  }

  /** @param {string[]} args */
  function parseRemainingOptions(args) {
    while (args.length) {
      const arg = args.shift();
      const value = args[0];
      if (arg !== undefined && value !== undefined && o.parse_opt(opt, o.opt_help, arg, value, "")) {
        args.shift();
      } else {
        return o.print_help("Unparsed option: " + arg);
      }
    }
  }

  /** @param {string[]} argv @param {MinerTestState} test */
  return function parseArgs(argv, test) {
    const args = argv.slice(2);
    const directive = args.shift();
    if (directive === undefined) {return o.print_help("No directive specified");}
    /** @type {Record<string, (args: string[]) => unknown>} */
    const parsers = {
      mine: parseMineArgs,
      test: (remaining) => { test.result_hash_hex = parseTestArgs(remaining); },
      bench: parseBenchArgs,
      algorithms: () => undefined,
    };
    const parser = Object.hasOwn(parsers, directive) ? parsers[directive] : null;
    if (!parser) {return o.print_help("Unknown directive " + directive);}
    parser(args);
    parseRemainingOptions(args);
    opt.job.algo = normalizeAlgoName(opt.job.algo) || null;
    return directive;
  };
};
