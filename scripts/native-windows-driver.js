"use strict";

/** One SDK-typed binding generator for all installed Windows GPU drivers. Own the linker
 * labels explicitly instead of decoding a compiler's private C++ name-mangling grammar.
 * The asm annotation names a symbol; it contains no assembly instructions or API prototypes.
 * @param {string[]} names @param {"hip" | "cuda"} kind */
function windowsDriverBindings(names, kind) {
  validateDriverNames(names, kind);
  const family = kind === "hip" ? "Hip" : "Cuda", prefix = "mom_" + kind;
  return [...(kind === "hip" ? ["#define __HIP_DISABLE_CPP_FUNCTIONS__"] : []),
    '#include "runtime/windows-' + kind + '-driver.h"', ...names.flatMap((name, index) => {
      const key = prefix + "_name_" + index, thunk = "Mom" + family + "Thunk" + index;
      return [`inline constexpr char ${key}[] = ${JSON.stringify(name)};`,
        `template<class> struct ${thunk};`,
        `template<class R, class... A> struct ${thunk}<R(*)(A...)> { static R call(A... args) asm("${prefix}_thunk_${index}"); };`,
        `template<class R, class... A> R ${thunk}<R(*)(A...)>::call(A... args) {`,
        `  return mom::runtime::Windows${family}Call<decltype(&::${name})>::template call<${key}>(args...);`,
        "}",
        `extern "C" { decltype(&::${name}) __imp_${name} = &${thunk}<decltype(&::${name})>::call; }`];
    }), ""].join("\n");
}

/** Metadata becomes source/linker text, so admit only unique bounded public driver names.
 * @param {string[]} names @param {"hip" | "cuda"} kind */
function validateDriverNames(names, kind) {
  if (!["hip", "cuda"].includes(kind) || !Array.isArray(names) || !names.length || names.length > 1024 ||
      new Set(names).size !== names.length || names.some(name => typeof name !== "string" ||
        !(kind === "hip" ? /^hip[A-Za-z0-9_]{1,127}$/ : /^(?:cu|nvml)[A-Z]\w{0,126}$/).test(name))) {
    throw new Error("Invalid Windows driver binding names");
  }
}

/** Generate typed static import bindings from the linker's actual HIP driver requirements.
 * No API prototype list, driver path or calling convention is duplicated here: the official
 * SDK supplies each signature, and the runtime fixes System32-only lookup. This metadata is a
 * trusted build input, never pool/miner configuration. @param {string} text */
function windowsHipBindings(text) {
  if (typeof text !== "string" || Buffer.byteLength(text) > (4 << 20) || /\bDelayImport\s*\{/.test(text)) {throw new Error("Invalid Windows HIP import metadata");}
  const blocks = [...text.matchAll(/^Import \{\r?\n\s*Name:\s*([^\s]+)\r?\n([\s\S]*?)^\}/gm)].filter(match => match[1]?.toLowerCase() === "amdhip64_7.dll");
  if (blocks.length !== 1) {throw new Error("Expected exactly one installed AMD HIP driver import");}
  /** @type {string[]} */
  const names = [];
  for (const line of (blocks[0]?.[2] || "").split(/\r?\n/).map(line => line.trim()).filter(Boolean)) {
    if (/^Import(?:LookupTable|AddressTable)RVA: 0x[0-9a-f]+$/i.test(line)) {continue;}
    const name = line.match(/^Symbol: (hip[A-Za-z0-9_]{1,127}) \(\d+\)$/)?.[1];
    if (!name || names.length >= 1024 || names.includes(name)) {throw new Error("Unsupported Windows HIP import symbol");}
    names.push(name);
  }
  if (!names.includes("hipGetDeviceCount")) {throw new Error("Windows HIP bindings must retain device discovery");}
  return {names, source: windowsDriverBindings(names, "hip")};
}

/** Some SDK objects call the C function directly; others use its __imp_ pointer. Bind both
 * to the same compiler-generated typed thunk, without hand-written API prototypes or assembly.
 * The recorded llvm-nm output must contain our exact generated linker labels.
 * @param {string} text @param {string[]} names */
function windowsHipAliases(text, names) {return windowsTypedDriverAliases(text, names, "hip");}

/** Share the same compiler-typed thunk/weak-alias mechanism for installed Windows drivers.
 * @param {string} text @param {string[]} names @param {"hip" | "cuda"} kind */
function windowsTypedDriverAliases(text, names, kind) {
  validateDriverNames(names, kind);
  const label = kind.toUpperCase(), prefix = "mom_" + kind + "_thunk_";
  if (typeof text !== "string" || Buffer.byteLength(text) > (4 << 20)) {throw new Error("Invalid typed " + label + " alias inputs");}
  const functions = text.split(/\r?\n/).map(line => line.match(/^(\S+) T(?:\s|$)/)?.[1]).filter(name => name?.startsWith(prefix));
  if (functions.length !== names.length) {throw new Error("Typed " + label + " thunk count mismatch");}
  return names.map((name, index) => {
    const symbol = prefix + index;
    if (functions.filter(value => value === symbol).length !== 1) {throw new Error("Unresolved official-SDK " + label + " thunk");}
    return "/alternatename:" + name + "=" + symbol;
  });
}

module.exports = {windowsDriverBindings, windowsHipBindings, windowsHipAliases, windowsTypedDriverAliases};
