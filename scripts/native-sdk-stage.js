"use strict";

const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");

/** Record exact argv and stream both output channels to a fresh stage log.
 * @param {string[]} command @param {string} cwd @param {NodeJS.ProcessEnv} env
 * @param {string} output @param {string} stage @param {string} label @param {number} timeout */
function runLoggedSdkStage(command, cwd, env, output, stage, label, timeout) {
  const prefix = path.join(output, "logs", stage), log = prefix + ".log";
  const fd = fs.openSync(log, "wx");
  try {
    fs.writeFileSync(prefix + ".json", JSON.stringify(command) + "\n", {flag: "wx"});
    const result = spawnSync(command[0] || "", command.slice(1), {cwd, env, stdio: ["ignore", fd, fd], timeout});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error(`${label} stage failed: ${stage}; inspect ${log}`, {cause: result.error});
    }
  } finally {
    fs.closeSync(fd);
  }
}

module.exports = {runLoggedSdkStage};
