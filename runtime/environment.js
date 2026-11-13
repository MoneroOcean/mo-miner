"use strict";

// Keep environment reads live and writes visible to subsequently-spawned native workers. A plain
// startup snapshot would silently lose Object.assign(process.env, workerEnv) in the real controller.
function environment() {
  /** @param {object} _target @param {string | symbol} name @param {unknown} value */
  function assign(_target, name, value) {
    if (typeof name !== "string" || typeof value === "symbol") {throw new TypeError("Cannot convert a Symbol value to a string");}
    globalThis.__momSetEnv(name, String(value));
    return true;
  }
  return new Proxy({}, {
    get(target, name) {
      return typeof name === "string" ? globalThis.__momGetEnv(name) ?? Reflect.get(target, name) : Reflect.get(target, name);
    },
    set: assign,
    deleteProperty(_target, name) {
      if (typeof name === "string") {globalThis.__momSetEnv(name, null);}
      return true;
    },
    has(target, name) {
      return typeof name === "string" && globalThis.__momGetEnv(name) !== undefined || Reflect.has(target, name);
    },
    ownKeys() {return globalThis.__momEnvKeys();},
    getOwnPropertyDescriptor(_target, name) {
      const value = typeof name === "string" ? globalThis.__momGetEnv(name) : undefined;
      return value === undefined ? undefined : {value, writable: true, enumerable: true, configurable: true};
    },
    defineProperty(_target, name, descriptor) {
      // Node's environment is not an ordinary mutable object: accessors/non-configurable keys
      // cannot be represented by the OS and would violate subsequent proxy enumeration invariants.
      if (!("value" in descriptor) || !descriptor.configurable || !descriptor.enumerable || !descriptor.writable) {
        throw Object.assign(new TypeError("Environment properties must be configurable, writable and enumerable values"),
          {code: "ERR_INVALID_OBJECT_DEFINE_PROPERTY"});
      }
      return assign(_target, name, descriptor.value);
    },
  });
}

module.exports = environment;
