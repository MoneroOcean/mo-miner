"use strict";

/** @param {string} arg @returns {string} */
function quoteWindowsCmdArg(arg) {
  if (/["%\r\n\0]/.test(arg)) {
    throw new Error("Windows cmd.exe cannot safely preserve this argument");
  }
  if (arg.length === 0) {return '""';}
  if (!/[\s&|<>()^]/.test(arg)) {return arg;}
  // CommandLineToArgvW treats backslashes immediately before a closing quote as escapes. Doubling
  // only that trailing run preserves a quoted path without changing ordinary path separators.
  return `"${arg.replace(/(\\+)$/, "$1$1")}"`;
}

/** @param {string[]} args @returns {string} */
function buildWindowsCmd(args) {
  return args.map(quoteWindowsCmdArg).join(" ");
}

/** @param {string[]} args @returns {string[]} */
function windowsCmdArgs(args) {
  return ["/d", "/v:off", "/s", "/c", `"${buildWindowsCmd(args)}"`];
}

module.exports = {buildWindowsCmd, quoteWindowsCmdArg, windowsCmdArgs};
