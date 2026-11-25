"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");

const stableFileTime = new Date(1600000000000);

/** Pin a file's atime/mtime so a later rewrite can restore them exactly. @param {string} file */
function setStableMtime(file) {fs.utimesSync(file, stableFileTime, stableFileTime);}

/**
 * Change a file's bytes, then restore its content and stable mtime: only ctime reveals the
 * replacement, as with a hostile swap during qualification. The pause outlasts coarse ctime clocks.
 * @param {string} file @returns {{mtimeNs: bigint, ctimeNs: bigint}}
 */
function transientRewriteRestore(file) {
  const original = fs.readFileSync(file), before = fs.statSync(file, {bigint: true}), changed = Buffer.from(original);
  changed[0] = (changed[0] || 0) ^ 0xff;
  fs.writeFileSync(file, changed);
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 25);
  fs.writeFileSync(file, original);
  setStableMtime(file);
  const after = fs.statSync(file, {bigint: true});
  assert.equal(after.mtimeNs, before.mtimeNs, "restore the original mtime");
  assert.notEqual(after.ctimeNs, before.ctimeNs, "the write must still advance ctime");
  return {mtimeNs: after.mtimeNs, ctimeNs: after.ctimeNs};
}

module.exports = {setStableMtime, transientRewriteRestore};
