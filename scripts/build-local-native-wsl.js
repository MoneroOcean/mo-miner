"use strict";

// Read-only Windows/WSL boundary checks. Actual build orchestration owns the lock,
// memory scope and producer lifecycle; these probes never install or build anything.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");

/** wsl.exe emits UTF-16LE to a redirected pipe on supported Windows versions.
 * @param {string | Buffer | null | undefined} value */
function wslText(value) {
  const text = Buffer.isBuffer(value) ? value.toString(value.includes(0) ? "utf16le" : "utf8") : value || "";
  if (text.length > 1 << 20 || Array.from(text).some(char =>
    char.charCodeAt(0) < 32 && !"\t\r\n".includes(char) || char.charCodeAt(0) === 127)) {throw new Error("Invalid WSL probe output");}
  return text.replace(/^\uFEFF/, "");
}

/** Do not guess a default when several developer distributions are installed. WSL names are
 * case-insensitive, so MOM_WSL_DISTRO selects the installed row's own spelling.
 * @param {string} names @param {string} versions @param {string} [requested] */
function selectWsl2Distribution(names, versions, requested = "") {
  const rows = names.split(/\r?\n/).map(row => row.trim()).filter(Boolean),
    match = rows.find(row => row.toLowerCase() === requested.toLowerCase());
  if (!rows.length || rows.length > 64 || new Set(rows.map(row => row.toLowerCase())).size !== rows.length ||
      rows.some(row => row.length > 128 || /^-/.test(row) || Array.from(row).some(char => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) ||
      requested && !match) {throw new Error("Select an installed WSL2 distribution with MOM_WSL_DISTRO");}
  if (!requested && rows.length !== 1) {throw new Error("Multiple WSL distributions installed; set MOM_WSL_DISTRO explicitly");}
  const selected = match || rows[0];
  if (!selected) {throw new Error("No WSL2 distribution selected");}
  const escaped = selected.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
    matching = versions.split(/\r?\n/).filter(row => new RegExp("^\\s*\\*?\\s*" + escaped + "\\s+.+?\\s+[12]\\s*$").test(row));
  if (matching.length !== 1 || !/\s2\s*$/.test(matching[0] || "")) {throw new Error("Selected distribution must use WSL2, not WSL1: " + selected);}
  return selected;
}

/** Prove one WSL2 runtime and round-trip the exact canonical checkout, using argv only.
 * The selected script path is derived from this checkout, never accepted from probe output.
 * @param {string} checkout @param {NodeJS.ProcessEnv} [environment]
 * @param {{platform?:NodeJS.Platform,arch?:string,spawn?:typeof spawnSync,realpath?:(file:string)=>string}} [dependencies] */
function prepareWindowsWslRelease(checkout, environment = process.env, dependencies = {}) {
  if ((dependencies.platform || process.platform) !== "win32" || (dependencies.arch || process.arch) !== "x64") {
    throw new Error("The Windows WSL release entrypoint requires Windows x64");
  }
  const realpath = dependencies.realpath || fs.realpathSync, spawn = dependencies.spawn || spawnSync,
    systemRoot = environment["SystemRoot"] || "C:\\Windows", executable = realpath(path.win32.join(systemRoot, "System32/wsl.exe"));
  if (!path.win32.isAbsolute(checkout) || /[\r\n\0]/.test(checkout) ||
      path.win32.normalize(checkout).toLowerCase() !== realpath(checkout).toLowerCase() ||
      executable.toLowerCase() !== path.win32.join(systemRoot, "System32/wsl.exe").toLowerCase()) {
    throw new Error("WSL release needs the canonical checkout and system wsl.exe");
  }
  /** @param {string[]} args */
  const query = args => {
    const result = spawn(executable, args, {env: environment, windowsHide: true, timeout: 30000, maxBuffer: 1 << 20, shell: false});
    if (result.error || result.signal || result.status !== 0) {throw new Error("WSL release preflight failed: " + args[0], {cause: result.error});}
    return wslText(result.stdout).trim();
  };
  const distribution = selectWsl2Distribution(query(["--list", "--quiet"]), query(["--list", "--verbose"]), environment["MOM_WSL_DISTRO"] || ""),
    prefix = ["--distribution", distribution, "--exec"],
    kernel = query([...prefix, "/usr/bin/uname", "-r"]);
  if (!/microsoft.*wsl2/i.test(kernel)) {throw new Error("Selected WSL runtime did not prove a WSL2 kernel");}
  const mapped = query([...prefix, "/usr/bin/wslpath", "-u", checkout]);
  if (!path.posix.isAbsolute(mapped) || path.posix.normalize(mapped) !== mapped || /[\r\n\0]/.test(mapped) ||
      path.win32.normalize(query([...prefix, "/usr/bin/wslpath", "-w", mapped])).toLowerCase() !== checkout.toLowerCase()) {
    throw new Error("WSL checkout path does not round-trip to the selected Windows checkout");
  }
  return {executable, distribution, checkout: mapped, prefix};
}

module.exports = {wslText, selectWsl2Distribution, prepareWindowsWslRelease};
