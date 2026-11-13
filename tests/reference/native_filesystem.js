"use strict";

const fs = require("node:fs"), path = require("node:path");

// One fixture runs unchanged through Node and the native compatibility layer. The driver creates
// separate write roots first, so one mode cannot inherit the other mode's permissions or files.
/** @param {string} root */
module.exports = function filesystemFixture(root) {
  const native = typeof globalThis.__momWriteFile === "function";
  const directory = path.join(root, native ? "native-writes" : "node-writes");
  const file = path.join(directory, "private-é-💎.json");
  const value = JSON.stringify({fixture: "é-💎", enabled: true});
  fs.writeFileSync(file, value, {mode: 0o600});
  fs.chmodSync(file, 0o640);
  const text = fs.readFileSync(file, "utf8");
  fs.writeFileSync(file, "{}", {mode: 0o600}); // Existing mode must survive a write, as under Node.
  const overwritten = fs.readFileSync(file, "utf8");
  const empty = path.join(directory, "empty");
  fs.writeFileSync(empty, "old content"); fs.writeFileSync(empty, "");
  const binary = path.join(directory, "binary");
  const data = Buffer.from([1, 0, 255, 128, 2]);
  fs.writeFileSync(binary, data.subarray(1, 4), {mode: "0600"});
  const bytes = fs.readFileSync(binary).toString("hex");
  fs.writeFileSync(binary, "00ff80", "hex");
  const encoded = fs.readFileSync(binary).toString("hex");
  /** @param {() => unknown} action */
  function code(action) {
    try {action(); return "NOT_REJECTED";} catch (error) {
      return error instanceof Error ? String(Reflect.get(error, "code")) : "unknown";
    }
  }
  const exclusive = code(() => fs.writeFileSync(file, "must not replace private config", {flag: "wx"}));
  const directoryError = code(() => fs.writeFileSync(directory, "bad"));
  const missingParent = code(() => fs.writeFileSync(path.join(directory, "absent/child"), "bad"));
  const chmodMissing = code(() => fs.chmodSync(path.join(directory, "absent"), 0o600));
  const invalid = code(() => fs.writeFileSync(file + "\0tail", "bad"));
  if (native) {
    // Validate native pointer/mode admission before open/truncation, not only the JS adapter.
    let rejected = 0;
    for (const mode of [-1, 1.5, NaN, Infinity, 0x100000000, "600", true, null, undefined, {}]) {
      try {Reflect.apply(globalThis.__momWriteFile, globalThis, [file, data, mode, false]);} catch {++rejected;}
      try {Reflect.apply(globalThis.__momChmod, globalThis, [file, mode]);} catch {++rejected;}
    }
    for (const flag of [0, 1, "true", null]) {
      try {Reflect.apply(globalThis.__momWriteFile, globalThis, [file, data, 0o600, flag]);} catch {++rejected;}
    }
    const forged = /** @type {Uint8Array} */ (/** @type {unknown} */ ({buffer: data.buffer, byteOffset: -1, byteLength: 4}));
    try {globalThis.__momWriteFile(file, forged, 0o600, false);} catch {++rejected;}
    if (rejected !== 25) {throw new Error("Native filesystem admitted malformed write/mode");}
  }
  if (fs.readFileSync(file, "utf8") !== "{}") {throw new Error("Failed write truncated private config");}
  return {text, overwritten, bytes, encoded, empty: fs.readFileSync(empty, "utf8"), exclusive,
    directoryError, missingParent, chmodMissing, invalid};
};
