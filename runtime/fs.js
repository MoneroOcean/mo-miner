"use strict";

const bytes = require("./bytes");

/** Filesystem APIs, unlike chdir, must reject NUL before calling C-string OS interfaces.
 * @param {unknown} value */
function filename(value) {
  if (Buffer.isBuffer(value)) {value = value.toString();}
  if (typeof value !== "string") {
    throw Object.assign(new TypeError("File path must be a string or Buffer"), {code: "ERR_INVALID_ARG_TYPE"});
  }
  if (value.includes("\0")) {
    throw Object.assign(new TypeError("File path contains NUL"), {code: "ERR_INVALID_ARG_VALUE"});
  }
  return value;
}

/** Preserve the public numeric/octal mode format; native code separately masks OS permission bits.
 * @param {unknown} value @param {number} [fallback] */
function fileMode(value, fallback) {
  value ??= fallback;
  if (typeof value === "string") {
    if (!/^[0-7]+$/.test(value)) {
      throw Object.assign(new TypeError("Invalid file mode"), {code: "ERR_INVALID_ARG_VALUE"});
    }
    value = Number.parseInt(value, 8);
  }
  if (typeof value !== "number") {throw Object.assign(new TypeError("File mode must be numeric or octal"), {code: "ERR_INVALID_ARG_TYPE"});}
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw Object.assign(new RangeError("File mode is outside uint32"), {code: "ERR_OUT_OF_RANGE"});
  }
  return value;
}

module.exports = {
  /** @param {unknown} file @param {BufferEncoding | {encoding?: BufferEncoding | null} | null} [options] */
  readFileSync(file, options) {
    const encoding = typeof options === "string" ? options : options?.encoding;
    const data = Buffer.from(globalThis.__momReadFile(filename(file)));
    return encoding ? data.toString(encoding) : data;
  },
  /** @param {unknown} file */
  existsSync(file) {
    try {return globalThis.__momFileExists(filename(file));} catch {return false;}
  },
  /** @param {unknown} directory */
  readdirSync(directory) {return globalThis.__momReadDirectory(filename(directory));},
  /** The controller uses ordinary private-file writes. Admit exclusive creation too, but fail
   * explicitly for unimplemented flags rather than accidentally overwriting an append/read file.
   * @param {unknown} file @param {unknown} data @param {unknown} [options] */
  writeFileSync(file, data, options) {
    const name = filename(file);
    const config = typeof options === "string" ? {encoding: options} : options ?? {};
    if (typeof config !== "object") {throw Object.assign(new TypeError("Invalid write options"), {code: "ERR_INVALID_ARG_TYPE"});}
    const encoding = Reflect.get(config, "encoding");
    if (encoding != null && (typeof encoding !== "string" || !Buffer.isEncoding(encoding))) {
      throw Object.assign(new TypeError("Unknown file encoding"), {code: "ERR_INVALID_ARG_VALUE"});
    }
    const mode = fileMode(Reflect.get(config, "mode"), 0o666);
    const flag = Reflect.get(config, "flag") ?? "w";
    if (flag !== "w" && flag !== "wx") {throw Object.assign(new TypeError("Unsupported write flag"), {code: "ERR_INVALID_ARG_VALUE"});}
    globalThis.__momWriteFile(name, bytes(data, encoding ?? undefined), mode, flag === "wx");
  },
  /** @param {unknown} file @param {unknown} mode */
  chmodSync(file, mode) {globalThis.__momChmod(filename(file), fileMode(mode));},
  /** @param {number} fd @param {unknown} data */
  writeSync(fd, data) {
    if (!Number.isInteger(fd) || fd < 0 || fd > 0x7fffffff) {
      throw Object.assign(new RangeError("Invalid file descriptor"), {code: "ERR_OUT_OF_RANGE"});
    }
    return globalThis.__momWrite(fd, bytes(data));
  },
};
