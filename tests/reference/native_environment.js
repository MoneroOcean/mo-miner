"use strict";

// Runs unchanged in direct Node and AOT mode. Never include the inherited host environment in
// transcripts: only temporary fixture variables are observed, and all original values are restored.
module.exports = function environmentChecks() {
  const prefix = "MOM_AOT_PARITY_";
  const names = ["TEXT", "NUMBER", "EMPTY", "UNDEFINED", "DEFINED", "DELETE", "NUL", "TRUNC"].map((name) => prefix + name);
  const saved = names.map((name) => process.env[name]);
  try {
    Object.assign(process.env, {[prefix + "TEXT"]: "é-💎", [prefix + "NUMBER"]: 42,
      [prefix + "EMPTY"]: "", [prefix + "UNDEFINED"]: undefined, [prefix + "DELETE"]: "gone"});
    Object.defineProperty(process.env, prefix + "DEFINED", {
      value: false, writable: true, enumerable: true, configurable: true,
    });
    process.env[prefix + "NUL"] = "before\0after";
    process.env[prefix + "TRUNC\0ignored"] = "truncated-name";
    delete process.env[prefix + "DELETE"];
    let invalidDescriptor = "";
    try {
      Object.defineProperty(process.env, prefix + "DELETE", {value: "must-not-appear"});
    } catch (error) {invalidDescriptor = /** @type {NodeJS.ErrnoException} */ (error).code || "missing";}
    const snapshot = {...process.env};
    return {values: Object.fromEntries(names.filter((name) => Object.hasOwn(snapshot, name))
      .map((name) => [name.slice(prefix.length), snapshot[name]])),
    present: prefix + "EMPTY" in process.env, deleted: prefix + "DELETE" in process.env,
    invalidDescriptor, inheritedMethod: typeof process.env.toString,
    descriptor: Object.getOwnPropertyDescriptor(process.env, prefix + "DEFINED")};
  } finally {
    names.forEach((name, index) => {
      const value = saved[index];
      if (value === undefined) {delete process.env[name];} else {process.env[name] = value;}
    });
  }
};
