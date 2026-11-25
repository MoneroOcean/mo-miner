"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

// Producer fixtures publish read-only SDK trees; restore owner write access so cleanup can unlink
// them. Symlinks are skipped so cleanup never changes modes outside the fixture.
/** @param {string} file */
function makeWritable(file) {
  if (!fs.existsSync(file)) {return;}
  const stat = fs.lstatSync(file);
  if (stat.isSymbolicLink()) {return;}
  if (stat.isDirectory()) {
    fs.chmodSync(file, 0o700);
    for (const name of fs.readdirSync(file)) {makeWritable(path.join(file, name));}
  } else if (stat.isFile()) {fs.chmodSync(file, 0o600);}
}

/** Create a private temporary directory removed when test `t` finishes.
 * @param {import("node:test").TestContext} t @param {string} prefix @returns {string} */
function tempDir(t, prefix) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  t.after(() => {makeWritable(root); fs.rmSync(root, {recursive: true, force: true});});
  return root;
}

module.exports = {tempDir};
