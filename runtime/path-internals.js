"use strict";

// Compile Node's maintained pure-JS path implementation, not a second C++ path algorithm. Only
// its small internal bootstrap surface is adapted. No Node engine or module loader is linked.
/** @param {Function} method */
const uncurry = (method) => Function.prototype.call.bind(method);
const primordials = {
  ArrayPrototypeIncludes: uncurry(Array.prototype.includes),
  ArrayPrototypeJoin: uncurry(Array.prototype.join),
  ArrayPrototypePush: uncurry(Array.prototype.push),
  ArrayPrototypeSlice: uncurry(Array.prototype.slice),
  FunctionPrototypeBind: uncurry(Function.prototype.bind),
  StringPrototypeCharCodeAt: uncurry(String.prototype.charCodeAt),
  StringPrototypeIncludes: uncurry(String.prototype.includes),
  StringPrototypeIndexOf: uncurry(String.prototype.indexOf),
  StringPrototypeLastIndexOf: uncurry(String.prototype.lastIndexOf),
  StringPrototypeRepeat: uncurry(String.prototype.repeat),
  StringPrototypeReplace: uncurry(String.prototype.replace),
  StringPrototypeSlice: uncurry(String.prototype.slice),
  StringPrototypeSplit: uncurry(String.prototype.split),
  StringPrototypeToLowerCase: uncurry(String.prototype.toLowerCase),
  StringPrototypeToUpperCase: uncurry(String.prototype.toUpperCase),
};

/** @param {unknown} value @param {string} name */
function invalidType(value, name) {
  return Object.assign(new TypeError(`Invalid ${name} argument (${typeof value})`), {code: "ERR_INVALID_ARG_TYPE"});
}

// Node 22 and 24 use different lazy glob modules. Neither is used by MoM: keep both
// names explicitly unsupported rather than importing a filesystem matcher or guessing.
function unsupportedGlob() {throw new Error("Native path.matchesGlob is not implemented");}

module.exports = {
  primordials,
  CHAR_UPPERCASE_A: 65, CHAR_LOWERCASE_A: 97, CHAR_UPPERCASE_Z: 90, CHAR_LOWERCASE_Z: 122,
  CHAR_DOT: 46, CHAR_FORWARD_SLASH: 47, CHAR_BACKWARD_SLASH: 92, CHAR_COLON: 58, CHAR_QUESTION_MARK: 63,
  /** @param {unknown} value @param {string} name */
  validateString(value, name) {if (typeof value !== "string") {throw invalidType(value, name);}},
  /** @param {unknown} value @param {string} name */
  validateObject(value, name) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {throw invalidType(value, name);}
  },
  isWindows: process.platform === "win32", isMacOS: process.platform === "darwin",
  /** @template T @param {() => T} factory */
  getLazy(factory) {
    /** @type {T} */
    let result;
    let initialized = false;
    return () => {
      if (!initialized) {result = factory(); initialized = true;}
      return result;
    };
  },
  minimatch: unsupportedGlob, matchGlobPattern: unsupportedGlob,
};
