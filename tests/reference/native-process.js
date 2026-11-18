"use strict";

// One fixture runs under Node and as AOT code. Never log the host's complete argv/environment.
module.exports = function processFixture() {
  const cwd = process.cwd();
  process.chdir("..");
  const changed = process.cwd() !== cwd;
  process.chdir(cwd);
  process.chdir(".\0ignored");
  const nulTruncates = process.cwd() === cwd;
  const directoryErrors = [];
  for (const value of [1, null, "bad\0path", "./__mom_missing_directory_for_chdir_test__"]) {
    try {
      process.chdir(/** @type {string} */ (value)); directoryErrors.push("NOT_REJECTED");
    } catch (error) {
      directoryErrors.push(error instanceof Error ? `${error.name}:${Reflect.get(error, "code")}` : "unknown");
    }
  }
  /** @type {Array<{value?: number, error?: string}>} */
  const statuses = [];
  const saved = process.exitCode;
  try {
    for (const value of [undefined, null, 0, -1, 2147483648, 4294967296, " 2 ", "1.5", "", false, 1n, NaN]) {
      try {
        Reflect.set(process, "exitCode", value);
        statuses.push(typeof process.exitCode === "number" ? {value: process.exitCode} : {});
      } catch (error) {
        statuses.push({error: error instanceof Error ? error.name : "unknown"});
      }
    }
  } finally {process.exitCode = saved;}
  return {
    args: process.argv.slice(2), platform: process.platform, arch: process.arch,
    pidValid: Number.isInteger(process.pid) && process.pid > 0,
    ppidValid: Number.isInteger(process.ppid) && process.ppid > 0,
    executableAbsolute: process.execPath.startsWith("/"), cwdRestored: process.cwd() === cwd,
    changed, nulTruncates, directoryErrors, statuses,
  };
};
