"use strict";

const fs = require("node:fs"), path = require("node:path");

/** Inventory immediate native-tool libraries, not target application libraries. Incomplete
 * SDK exports can retain dangling aliases; ignore only those aliases, never loader payloads,
 * escaping links, permission failures or link cycles. Callers still probe every required tool.
 * @param {string[]} roots @param {string} label @returns {string[]} */
function nativeRuntimeLibraries(roots, label) {
  if (!roots.length) {throw new Error(label + " native LLVM runtime roots are empty");}
  const files = [];
  for (const directory of roots) {
    const stat = fs.lstatSync(directory, {throwIfNoEntry: false});
    if (!stat?.isDirectory() || fs.realpathSync(directory) !== directory) {
      throw new Error(label + " native LLVM runtime root is not a canonical directory: " + directory);
    }
    const entries = fs.readdirSync(directory, {withFileTypes: true}).filter(entry => /\.so(?:\.\d+)*$/.test(entry.name));
    // A canonical SDK lib directory may be empty when its tools use only OS loader libraries.
    if (entries.length > 4096) {throw new Error(label + " native LLVM runtime root exceeds the shared-library bound: " + directory);}
    const root = fs.realpathSync(directory) + path.sep, before = files.length;
    for (const entry of entries) {
      const file = path.join(directory, entry.name);
      let target;
      try {target = fs.realpathSync(file);} catch (error) {
        if (entry.isSymbolicLink() && error instanceof Error && /** @type {NodeJS.ErrnoException} */ (error).code === "ENOENT" &&
            fs.lstatSync(file, {throwIfNoEntry: false})?.isSymbolicLink()) {continue;}
        throw error;
      }
      if ((!entry.isFile() && !entry.isSymbolicLink()) || !target.startsWith(root) || !fs.statSync(file).isFile()) {
        throw new Error(label + " native LLVM runtime library escapes its selected root: " + file);
      }
      files.push(file);
    }
    if (entries.length && files.length === before) {throw new Error(label + " native LLVM runtime root has no available shared libraries: " + directory);}
  }
  return files;
}

module.exports = {nativeRuntimeLibraries};
