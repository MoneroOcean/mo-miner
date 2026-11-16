"use strict";

// Both CI and local builds use this exact pinned host tool; it is never distributed.
const fs = require("node:fs"), path = require("node:path"), {spawnSync} = require("node:child_process");
const {fileHash} = require("./build-source-inputs");
const tool = require("./native-packing-tool.json");

/** Read only the catalog's exact canonical regular file, without following links.
 * @param {string} file @param {{bytes:number,sha256:string}} pin */
function admitPackingFile(file, pin) {
  const stat = fs.lstatSync(file);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.nlink !== 1 || fs.realpathSync(file) !== file ||
      stat.size !== pin.bytes || fileHash(file) !== pin.sha256) {throw new Error("Pinned packing tool bytes differ: " + file);}
}

/** Fetch just the official pinned archive and extract one member to an exclusive file.
 * Unknown/partial directories are preserved, not overwritten or silently re-admitted.
 * @param {string} output @param {NodeJS.ProcessEnv} [env] @param {typeof spawnSync} [spawn] */
function prepareNativePackingTool(output, env = process.env, spawn = spawnSync) {
  if (process.platform !== "linux" || process.arch !== "x64") {throw new Error("The pinned packing host tool requires Linux x64");}
  if (!path.isAbsolute(output) || path.resolve(output) !== output || /[\0\r\n]/.test(output) ||
      fs.realpathSync(path.dirname(output)) !== path.dirname(output)) {throw new Error("Packing tool needs a canonical existing parent");}
  if (!/^\d+\.\d+\.\d+$/.test(tool.version) || tool.platform !== "linux" || tool.architecture !== "x64" ||
      tool.archive.url !== `https://github.com/upx/upx/releases/download/v${tool.version}/upx-${tool.version}-amd64_linux.tar.xz` ||
      tool.executable.member !== `upx-${tool.version}-amd64_linux/upx` ||
      ![tool.archive, tool.executable].every(row => Number.isSafeInteger(row.bytes) && row.bytes > 0 && row.bytes < (16 << 20) && /^[a-f0-9]{64}$/.test(row.sha256))) {
    throw new Error("Unexpected pinned native packing tool catalog");
  }
  const archive = path.join(output, "upx.tar.xz"), executable = path.join(output, "upx"),
    stat = fs.lstatSync(output, {throwIfNoEntry: false});
  if (stat) {
    if (!stat.isDirectory() || stat.isSymbolicLink() || fs.realpathSync(output) !== output ||
        fs.readdirSync(output).sort().join(",") !== "upx,upx.tar.xz") {
      throw new Error("Packing tool cache is unknown or incomplete; preserve it for diagnosis");
    }
    admitPackingFile(archive, tool.archive); admitPackingFile(executable, tool.executable);
    fs.accessSync(executable, fs.constants.X_OK);
    return {upx: executable, upxSha256: tool.executable.sha256};
  }
  fs.mkdirSync(output, {mode: 0o700});
  /** @param {string} command @param {string[]} args @param {number} timeout @param {number} [stdout] */
  const run = (command, args, timeout, stdout) => {
    const result = spawn(command, args, {env: {...env, TAR_OPTIONS: "", LC_ALL: "C"}, timeout,
      maxBuffer: 2 << 20, stdio: ["ignore", stdout ?? "pipe", "pipe"]});
    if (result.error || result.signal || result.status !== 0) {
      throw new Error(`Pinned packing tool acquisition failed: ${command} (exit ${result.status}, signal ${result.signal || "none"})`, {cause: result.error});
    }
  };
  run("curl", ["--disable", "--fail", "--location", "--proto", "=https", "--proto-redir", "=https", "--max-redirs", "5",
    "--max-time", "120", "--max-filesize", String(tool.archive.bytes), "--output", archive, tool.archive.url], 125000);
  admitPackingFile(archive, tool.archive);
  const fd = fs.openSync(executable, "wx", 0o700);
  try {run("tar", ["-xOf", archive, "--", tool.executable.member], 30000, fd);} finally {fs.closeSync(fd);}
  admitPackingFile(executable, tool.executable); admitPackingFile(archive, tool.archive);
  return {upx: executable, upxSha256: tool.executable.sha256};
}

module.exports = {admitPackingFile, prepareNativePackingTool};
